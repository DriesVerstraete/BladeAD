"""Load + validate a case directory.

A *case* is a directory containing a `case.py` module with a module-level
`CASE = {...}` dict (a Python module, not JSON/YAML -- so it can carry computed
values like ISA density or a cosine station array, and matches the workspace
no-argparse style). The same directory is the output directory: every pkl, log,
plot and `_anchor_registry.json` is written back into it.

`Context` bundles the resolved case dict with its directory so the sweep layer
never has to re-load or re-derive paths.

The `CASE` schema (see `cases/shahjahan_proprotor/case.py` for a worked example):

    rotor      : n_blades, radius_m, hub_radius_m, n_chord_cps, n_twist_cps,
                 bspline_order, radial_distribution ("cosine"|"uniform"), num_radial
    operating  : hover  {altitude_m, airspeed_m_s, thrust_n}
                 cruise {altitude_m, airspeed_m_s, thrust_n}
    bounds     : chord_m, twist_deg, hover_rpm, cruise_rpm,
                 hover_pitch_deg, cruise_pitch_deg          (each a (lo, hi) pair)
    constraints: cl_max, taper (lo, hi), thrust_tolerance_n
    airfoil    : section_boundaries_r_over_r, names, t_over_c, table_dir
    acoustic   : n_rotors, cruise_noise_cap_db
    seed       : either a dict {chord_cps_m, twist_cps_deg, hover_rpm, cruise_rpm,
                 hover_pitch_deg, cruise_pitch_deg}  OR the string "inverse-design"
    objectives : list of objective-slot specs, in slot order -- either plain
                 name strings ("fm_hover"/"eta_cruise"/"hover_noise", mapped to
                 their canonical specs) or dicts
                 {name, result_key, goal, role[, proxy_min_key, label]}.
                 Exactly one slot must have role "proxy". See `parse_objectives`.
    sweep      : n_points, n_points_3obj, maxiter   (ftol optional, default 1e-6;
                 grid_fm / grid_noise optional, default 8 -- the epsilon-grid
                 3-objective lattice resolution)
    cruise_anchor_fm_floor_frac : float (default 0.65)   -- only used by the
                 legacy `cruise-only` mode; the working sweep never needs it.
"""
import importlib.util
import os
import re
from dataclasses import dataclass

import numpy as np

_REQUIRED_TOP = ("rotor", "operating", "bounds", "constraints", "airfoil",
                 "acoustic", "seed", "objectives", "sweep")

# --- objective slots -------------------------------------------------------
#
# The machinery (sweep_common / edges / grid / nobj / orchestrator / plot)
# knows only *slots* f[0], f[1], ...  Objective identity -- which physical
# result quantity a slot is, whether it is maximised or minimised, and how the
# solver handles it -- lives here (the spec) and in solve.py (the BEM-output
# reads).  See briefs/rotor-pareto-objective-slots-refactor-brief.md.
#
#   name         : identifier -- logs, pkl-tag keys, plot labels, registry keys
#   result_key   : dotted path into solve.py's result dict (the REPORTED value)
#   goal         : "max" | "min"
#   role         : "proxy" | "floor" | "cap" | "reserved"
#                  proxy    -- the objective the solver optimises directly
#                  floor    -- enforced as  result_key >= epsilon   (natural for max)
#                  cap      -- enforced as  result_key <= epsilon   (natural for min)
#                  reserved -- a slot with no solver code yet (skipped w/ a note)
#   proxy_min_key: proxy only -- the (well-conditioned) quantity the solver
#                  actually minimises as a stand-in.  May equal result_key.
#   label        : plot axis label (default: name)

_RESERVED_OBJECTIVE_NAMES = ("electrical_power",)     # step 4 slot, no solver code yet

# back-compat: the three names the frozen scripts / archived cases used as a
# plain string list map to these canonical specs.
_CANONICAL_SPECS = {
    "fm_hover":    {"name": "fm_hover",    "result_key": "figure_of_merit",
                    "goal": "max", "role": "floor", "label": "hover FM"},
    "eta_cruise":  {"name": "eta_cruise",  "result_key": "cruise_efficiency",
                    "goal": "max", "role": "proxy", "proxy_min_key": "cruise_power",
                    "label": "cruise efficiency"},
    "hover_noise": {"name": "hover_noise", "result_key": "acoustics.hover_ospl_db",
                    "goal": "min", "role": "cap", "label": "hover noise (dB)"},
    "electrical_power": {"name": "electrical_power", "result_key": "electrical_power",
                         "goal": "min", "role": "reserved", "label": "electrical power (W)"},
}

_EPS_TAG_RE = re.compile(r"_eps-([A-Za-z0-9_]+)=(-?\d+(?:\.\d+)?)")


@dataclass
class ObjectiveSpec:
    name: str
    result_key: str
    goal: str
    role: str
    proxy_min_key: str = None
    label: str = None

    def __post_init__(self):
        if not self.label:
            self.label = self.name
        if self.role == "proxy" and not self.proxy_min_key:
            self.proxy_min_key = self.result_key

    @property
    def is_acoustic(self):
        return self.result_key.startswith("acoustics.")


def parse_objectives(raw):
    """Normalise `case['objectives']` (a list of plain-string names OR a list of
    spec dicts) into a validated list of `ObjectiveSpec`, in slot order.

    Rules: >= 2 objectives; exactly one `proxy` among the non-reserved slots;
    every non-proxy non-reserved slot has a role; `result_key` non-empty;
    `proxy_min_key` present on the proxy (defaults to its `result_key`)."""
    specs = []
    for entry in raw:
        if isinstance(entry, str):
            if entry not in _CANONICAL_SPECS:
                raise ValueError(
                    f"unknown objective name {entry!r}; known: {sorted(_CANONICAL_SPECS)}")
            entry = _CANONICAL_SPECS[entry]
        d = dict(entry)
        if "name" not in d or "result_key" not in d:
            raise ValueError(f"objective spec needs 'name' and 'result_key': {d!r}")
        d.setdefault("goal", "max")
        if d["goal"] not in ("max", "min"):
            raise ValueError(f"objective {d['name']}: goal must be 'max'|'min', got {d['goal']!r}")
        d.setdefault("role", "cap" if d["goal"] == "min" else "floor")
        if d["role"] not in ("proxy", "floor", "cap", "reserved"):
            raise ValueError(f"objective {d['name']}: bad role {d['role']!r}")
        specs.append(ObjectiveSpec(
            name=d["name"], result_key=d["result_key"], goal=d["goal"], role=d["role"],
            proxy_min_key=d.get("proxy_min_key"), label=d.get("label")))

    live = [s for s in specs if s.role != "reserved"]
    if len(live) < 2:
        raise ValueError(f"need >= 2 live objectives, got {[s.name for s in live]}")
    n_proxy = sum(s.role == "proxy" for s in live)
    if n_proxy != 1:
        raise ValueError(f"need exactly 1 'proxy' objective, got {n_proxy} "
                         f"({[s.name for s in live if s.role == 'proxy']})")
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate objective names: {names}")
    return specs


def epsilon_tag(epsilons):
    """`_eps-<name>=<value>` per active epsilon-constraint, sorted -- the pkl
    filename fragment that lets `_existing_node` / `nearest_warm` recognise a
    solved node. Round to 4 dp for a stable filename."""
    return "".join(f"_eps-{k}={float(v):.4f}" for k, v in sorted(epsilons.items()))


def parse_epsilon_tag(filename):
    """Inverse of `epsilon_tag`: {name: value} parsed from a pkl filename."""
    return {m.group(1): float(m.group(2)) for m in _EPS_TAG_RE.finditer(filename)}


def load_case_dict(case_dir):
    """Import `<case_dir>/case.py` and return its `CASE` dict (validated)."""
    case_py = os.path.join(case_dir, "case.py")
    if not os.path.isfile(case_py):
        raise FileNotFoundError(f"no case.py in {case_dir!r}")
    spec = importlib.util.spec_from_file_location("_rotor_pareto_case", case_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CASE"):
        raise AttributeError(f"{case_py} has no module-level CASE dict")
    case = dict(module.CASE)
    _validate(case, case_dir)
    return case


def _validate(case, case_dir):
    missing = [k for k in _REQUIRED_TOP if k not in case]
    if missing:
        raise ValueError(f"case is missing required keys: {missing}")

    rd = case["rotor"].get("radial_distribution", "uniform")
    if rd not in ("cosine", "uniform"):
        raise ValueError(f"rotor.radial_distribution must be 'cosine' or 'uniform', got {rd!r}")

    for pt in ("hover", "cruise"):
        entry = case["operating"].get(pt)
        if entry is None or not {"altitude_m", "airspeed_m_s", "thrust_n"} <= set(entry):
            raise ValueError(f"operating.{pt} needs altitude_m, airspeed_m_s, thrust_n")

    parse_objectives(case["objectives"])          # raises on a malformed spec / bad proxy count

    af = case["airfoil"]
    tbl = af.get("table_dir")
    if not tbl or not os.path.isdir(tbl):
        raise ValueError(f"airfoil.table_dir {tbl!r} is not a directory")
    if len(af["section_boundaries_r_over_r"]) != len(af["names"]) + 1:
        raise ValueError("airfoil.section_boundaries_r_over_r must have one more entry than names")
    if len(af["t_over_c"]) != len(af["names"]):
        raise ValueError("airfoil.t_over_c must match airfoil.names in length")

    seed = case["seed"]
    if isinstance(seed, str):
        if seed != "inverse-design":
            raise ValueError(f"seed string must be 'inverse-design', got {seed!r}")
    elif not isinstance(seed, dict):
        raise ValueError("seed must be a dict or the string 'inverse-design'")

    for k in ("n_points", "n_points_3obj", "maxiter"):
        if k not in case["sweep"]:
            raise ValueError(f"sweep.{k} is required")

    case.setdefault("cruise_anchor_fm_floor_frac", 0.65)
    case["sweep"].setdefault("ftol", 1e-6)
    case["sweep"].setdefault("grid_fm", 8)
    case["sweep"].setdefault("grid_noise", 8)


def active_objectives(case):
    """Ordered list of the live (non-reserved) `ObjectiveSpec`s this case sweeps."""
    return [s for s in parse_objectives(case["objectives"]) if s.role != "reserved"]


def reserved_objectives(case):
    """Ordered list of reserved `ObjectiveSpec`s (no solver code yet)."""
    return [s for s in parse_objectives(case["objectives"]) if s.role == "reserved"]


def proxy_objective(case):
    return next(s for s in active_objectives(case) if s.role == "proxy")


# --- shared numeric helpers (pure; used by solve.py and seed.py) -------------

def isa_atmos_density(altitude_m):
    """ISA troposphere air density (kg/m^3), valid to 11 km."""
    temperature = 288.15 - 0.0065 * altitude_m
    pressure = 101325.0 * (temperature / 288.15) ** 5.25588
    return pressure / (287.058 * temperature)


def cosine_stations(n):
    """n Chebyshev cell-centred nodes on the open interval (0, 1):
    x_i = 0.5*(1 - cos(pi*(i+0.5)/n)). Clustered at both ends, never touching
    0 or 1 -- matches BladeAD's element-centre convention and the fork's
    non-uniform-grid validator."""
    i = np.arange(n)
    return 0.5 * (1.0 - np.cos(np.pi * (i + 0.5) / n))


@dataclass
class Context:
    """A resolved case + its directory (which is also the output directory)."""
    case_dir: str
    case: dict

    @classmethod
    def load(cls, case_dir):
        case_dir = os.path.abspath(case_dir)
        return cls(case_dir=case_dir, case=load_case_dict(case_dir))

    @property
    def registry_path(self):
        return os.path.join(self.case_dir, "_anchor_registry.json")

    @property
    def sweep(self):
        return self.case["sweep"]

    def out(self, name):
        return os.path.join(self.case_dir, name)
