"""Option D -- acoustic sweep of the NSGA-II scout population.

The scout (`scout_run.py`) maps the hover_elec x cruise_elec basin structure but
is acoustically blind (the BEM kernel cannot evaluate noise). This script takes
each converged scout individual, pins its blade, and does a single acoustic
evaluation (`solve.run` with maxiter=1 -- the `maxiter` hook on `solve.run`),
reading back the cruise vehicle OSPL. The question it answers: is any
hover/cruise-electric basin *also* quiet, i.e. is there a low-noise design the
electric-power-only scout would have walked past?

Not an optimisation. One graph build + ~one BEM+acoustic eval per individual,
`N_WORKERS` in parallel.

Direct invocation (NOT `conda run`, NOT `| tail`), AFTER `scout_run.py`:

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/acoustic_scout_sweep.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/acoustic_scout_sweep.log 2>&1
"""
import csv
import os
import pickle
import subprocess
import sys
import time

import numpy as np

N_WORKERS = 4
HYBRID_FRONT_MIN_DB = 73.0          # the hybrid noise front's bracketed minimum (dry run: ~73-77 dB)
BASIN_MARGIN_DB = 2.0               # "quieter basin" = more than this below HYBRID_FRONT_MIN_DB

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCOUT_PKL = os.path.join(_HERE, "nsga2_result_hybrid_scout.pkl")
_SEED_TMP = os.path.join(_HERE, "_acoustic_sweep_seeds")
_CSV = os.path.join(_HERE, "acoustic_scout_sweep.csv")
_PLOT = os.path.join(_HERE, "acoustic_scout_sweep.png")

_FIELDS = ["idx", "cruise_noise_db", "hover_elec_W", "cruise_elec_W",
           "all_nonacoustic_constraints_ok", "failed_constraints", "status"]


def _evaluate(args):
    """One scout individual -> its acoustic row. Runs solve in a subprocess for
    csdl-recorder isolation; keeps the row even if non-acoustic constraints fail
    (a pinned scout blade is not re-trimmed)."""
    idx, var_dict = args
    from BladeAD.optimisation.sweep_common import nsga_geometry_to_seed

    seed_path = os.path.join(_SEED_TMP, f"seed_{idx:03d}.pkl")
    nsga_geometry_to_seed(var_dict, seed_path, fixed_pitch=True)

    code = (
        "from BladeAD.optimisation import solve; "
        f"solve.run_case_dir(r'{_HERE}', optimize='cruise_noise', "
        f"pin_geometry_from=r'{seed_path}', with_acoustics=True, maxiter=1)"
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, "-c", code], cwd=_HERE,
                          capture_output=True, text=True, env=env)
    saved = [ln for ln in proc.stdout.splitlines() if ln.startswith("Saved to ")]
    if proc.returncode != 0 or not saved:
        return {"idx": idx, "cruise_noise_db": np.nan, "hover_elec_W": np.nan,
                "cruise_elec_W": np.nan, "all_nonacoustic_constraints_ok": False,
                "failed_constraints": "", "status": f"solve_failed_exit_{proc.returncode}"}

    with open(saved[-1][len("Saved to "):].strip(), "rb") as f:
        data = pickle.load(f)
    ov = data.get("objective_values", {})
    checks = data.get("constraint_checks", {})
    nonac = {k: v for k, v in checks.items() if "noise" not in k.lower()}
    bad = [k for k, ok in nonac.items() if not ok]
    return {"idx": idx,
            "cruise_noise_db": float(ov.get("cruise_noise", np.nan)),
            "hover_elec_W": float(ov.get("hover_elec", np.nan)),
            "cruise_elec_W": float(ov.get("cruise_elec", np.nan)),
            "all_nonacoustic_constraints_ok": not bad,
            "failed_constraints": ";".join(bad),
            "status": "ok"}


def main():
    with open(_SCOUT_PKL, "rb") as f:
        scout = pickle.load(f)
    pop = scout.get("final_population") or []
    if not pop:
        raise SystemExit(f"{_SCOUT_PKL} has no final_population -- run scout_run.py first")
    os.makedirs(_SEED_TMP, exist_ok=True)
    jobs = [(i, row["var_dict"]) for i, row in enumerate(pop)]
    print(f"[{time.strftime('%H:%M:%S')}] acoustic sweep: {len(jobs)} scout individuals, "
          f"{N_WORKERS} workers", flush=True)

    from multiprocess import Pool
    rows = []
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        for row in pool.imap_unordered(_evaluate, jobs):
            rows.append(row)
            print(f"[{time.strftime('%H:%M:%S')}] {len(rows):3d}/{len(jobs)}  "
                  f"idx {row['idx']:3d}  noise {row['cruise_noise_db']:.2f} dB  "
                  f"{row['status']}"
                  f"{'  (' + row['failed_constraints'] + ')' if row['failed_constraints'] else ''}",
                  flush=True)
            with open(_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_FIELDS)
                w.writeheader()
                w.writerows(sorted(rows, key=lambda r: r["idx"]))

    rows.sort(key=lambda r: r["idx"])
    valid = [r for r in rows if np.isfinite(r["cruise_noise_db"])]
    feas = [r for r in valid if r["all_nonacoustic_constraints_ok"]]
    print(f"\n[{time.strftime('%H:%M:%S')}] done in {(time.time() - t0) / 60:.1f} min  "
          f"({len(valid)}/{len(rows)} evaluated, {len(feas)} non-acoustically feasible)", flush=True)
    if valid:
        pool_min = min(valid, key=lambda r: r["cruise_noise_db"])
        print(f"quietest scout individual: idx {pool_min['idx']}  "
              f"{pool_min['cruise_noise_db']:.2f} dB  "
              f"(feasible={pool_min['all_nonacoustic_constraints_ok']})", flush=True)
        threshold = HYBRID_FRONT_MIN_DB - BASIN_MARGIN_DB
        quiet = [r for r in feas if r["cruise_noise_db"] < threshold]
        if quiet:
            print(f"\n*** BASIN: {len(quiet)} feasible scout individual(s) below "
                  f"{threshold:.1f} dB (hybrid front min {HYBRID_FRONT_MIN_DB} - "
                  f"{BASIN_MARGIN_DB} margin) -- the scout found a quiet basin the "
                  f"hybrid missed. DVs:", flush=True)
            for r in sorted(quiet, key=lambda r: r["cruise_noise_db"]):
                vd = pop[r["idx"]]["var_dict"]
                print(f"  idx {r['idx']}: {r['cruise_noise_db']:.2f} dB  {vd}", flush=True)
        else:
            print(f"\nNO BASIN: no feasible scout individual is more than "
                  f"{BASIN_MARGIN_DB} dB below the hybrid front min "
                  f"({HYBRID_FRONT_MIN_DB} dB). The hybrid front is not missing a "
                  f"low-noise region.", flush=True)

    _plot(rows)
    print(f"csv -> {_CSV}\nplot -> {_PLOT}", flush=True)


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = [r for r in rows if np.isfinite(r["cruise_noise_db"])
         and np.isfinite(r["cruise_elec_W"])]
    if not v:
        return
    ce = np.array([r["cruise_elec_W"] for r in v])
    cn = np.array([r["cruise_noise_db"] for r in v])
    he = np.array([r["hover_elec_W"] for r in v])
    feas = np.array([r["all_nonacoustic_constraints_ok"] for r in v])

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(ce[feas], cn[feas], c=he[feas], cmap="viridis", s=45,
                    label="non-acoustically feasible")
    ax.scatter(ce[~feas], cn[~feas], facecolors="none", edgecolors="0.6", s=45,
               label="infeasible (pinned, not re-trimmed)")
    ax.axhspan(HYBRID_FRONT_MIN_DB, 77.0, color="C1", alpha=0.15,
               label="hybrid noise front (~73-77 dB)")
    ax.axhline(HYBRID_FRONT_MIN_DB - BASIN_MARGIN_DB, color="C3", ls="--", lw=1,
               label=f"basin threshold ({HYBRID_FRONT_MIN_DB - BASIN_MARGIN_DB:.0f} dB)")
    fig.colorbar(sc, label="hover electrical power (W)")
    ax.set_xlabel("cruise electrical power (W)")
    ax.set_ylabel("cruise vehicle OSPL (dB)")
    ax.set_title("Option D -- acoustic sweep of the hover/cruise-elec scout")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_PLOT, dpi=130)


if __name__ == "__main__":
    main()
