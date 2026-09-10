"""Step 1 of the noise BASIN-SCOUT plan -- reconcile the cruise-OSPL discrepancy
before building the scout.

The hybrid noise front floors at cruise_noise = 72.75 dB, but earlier smokes /
single solves reported ~62-64.7 dB "for this geometry" (rotor-optimisation
status.md). Question: is ~62 dB a real converged cruise OSPL that the whole
epsilon/tracer pipeline is basin-trapped away from, or an artefact
(case-default cap literal / a different VPP case / an unconverged solve)?

LIKELY ANSWER, already visible in the stored front data (confirm, don't assume):
the quiet-end pkl's `acoustics` block has cruise_single_ospl_db = 63.72 and
cruise_vehicle_ospl_db = 72.75 -- difference = 10*log10(8) = 9.03 dB, the
8-rotor vehicle sum. So "~62-64.7 dB for this geometry" is the SINGLE-ROTOR
OSPL and 72.75 dB is the SAME solve's vehicle OSPL; not an unconverged solve,
not a missed basin. This script confirms that holds under a fresh converged
solve and prints both numbers side by side.

This runs a FULLY CONVERGED acoustic solve (case sweep maxiter, NOT maxiter=1)
on a set of pinned blade geometries native to this FPP noise case and reads
back the converged cruise + hover OSPL. Blade chord/twist/hover_rpm/collective
are frozen (pin_geometry_from); cruise_rpm retrims to hold cruise thrust; the
acoustic graph is built and read back.

Each solve runs in a subprocess (csdl-recorder isolation), same pattern as
acoustic_scout_sweep.py. Results -> reconcile_quiet_end.json + stdout table.

Run (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/reconcile_quiet_end.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/reconcile_quiet_end.log 2>&1

~5 acoustic solves, ~300 s each cold / faster warm -> budget ~15-25 min.
"""
import json
import os
import pickle
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

# (label, geometry pkl relative to _HERE, note). All are FPP-native pkls with
# keys pin_geometry_from reads: chord_cps, twist_cps (rad), rpm, cruise_rpm,
# pitch_deg / hover_pitch_deg.
TARGETS = [
    ("quiet_end_front_point",
     "stage1_opt-cruise_elec_grid-cosine32_ac_eps-cruise_noise=72.7488_eps-hover_elec=22565.0104.pkl",
     "the 72.75 dB / 22.57 kW hover quiet-end ND point of the hybrid noise front"),
    ("loud_end_front_point",
     "stage1_opt-cruise_elec_grid-cosine32_ac_eps-cruise_noise=77.3735_eps-hover_elec=19514.8203.pkl",
     "a loud-end ND point (~77.4 dB, min hover_elec pin) for contrast"),
    ("fpp_cold_inverse_design_seed",
     "seed_inverse_design_fpp.pkl",
     "the cold FPP inverse-design blend seed (Blocker-1 smoke reported 72.0 dB)"),
    ("cruise_noise_single_obj_optimum",
     "stage1_opt-cruise_noise_grid-cosine32.pkl",
     "the standalone min-cruise_noise anchor solve (auto-bracket lower anchor)"),
]


def _run_one(label, pkl_rel, note):
    pkl_abs = os.path.join(_HERE, pkl_rel)
    if not os.path.exists(pkl_abs):
        return {"label": label, "status": f"missing_geometry:{pkl_rel}"}
    code = (
        "import pickle; from BladeAD.optimisation import solve; "
        f"r = solve.run_case_dir(r'{_HERE}', pin_geometry_from=r'{pkl_abs}', "
        "with_acoustics=True); "
        "print('RESULT_PKL', r if isinstance(r, str) else '')"
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-u", "-c", code], cwd=_HERE,
                          capture_output=True, text=True, env=env)
    dt = time.time() - t0
    saved = [ln[len("Saved to "):].strip()
             for ln in proc.stdout.splitlines() if ln.startswith("Saved to ")]
    if proc.returncode != 0 or not saved:
        return {"label": label, "note": note, "status": f"solve_failed_exit_{proc.returncode}",
                "wall_s": dt,
                "stderr_tail": "\n".join(proc.stderr.splitlines()[-25:]),
                "stdout_tail": "\n".join(proc.stdout.splitlines()[-25:])}
    with open(saved[-1], "rb") as f:
        d = pickle.load(f)
    ac = d.get("acoustics") or {}
    return {
        "label": label, "note": note, "status": "ok", "wall_s": dt,
        "result_pkl": os.path.basename(saved[-1]),
        "cruise_vehicle_ospl_db": ac.get("cruise_vehicle_ospl_db"),
        "cruise_single_ospl_db": ac.get("cruise_single_ospl_db"),
        "cruise_tonal_spl_db": ac.get("cruise_tonal_spl_db"),
        "cruise_broadband_spl_db": ac.get("cruise_broadband_spl_db"),
        "hover_ospl_db": ac.get("hover_ospl_db"),
        "cruise_noise_cap_db": ac.get("cruise_noise_cap_db"),
        "n_rotors": ac.get("n_rotors"),
        "objective_values": d.get("objective_values"),
        "cruise_rpm_retrimmed": d.get("cruise_rpm"),
        "hover_rpm": d.get("rpm"),
        "collective_deg": d.get("hover_pitch_deg", d.get("pitch_deg")),
        "cruise_thrust": d.get("cruise_thrust"),
        "hover_thrust": d.get("hover_thrust"),
        "cruise_stall_ratio": d.get("cruise_stall_ratio"),
        "hover_stall_ratio": d.get("hover_stall_ratio"),
        "constraint_checks": {k: bool(v) for k, v in (d.get("constraint_checks") or {}).items()},
    }


def main():
    print(f"[{time.strftime('%H:%M:%S')}] reconcile_quiet_end: {len(TARGETS)} converged "
          f"acoustic solves\n", flush=True)
    rows = []
    for label, pkl_rel, note in TARGETS:
        print(f"[{time.strftime('%H:%M:%S')}] --> {label}  ({note})", flush=True)
        row = _run_one(label, pkl_rel, note)
        rows.append(row)
        cn = row.get("cruise_vehicle_ospl_db")
        print(f"[{time.strftime('%H:%M:%S')}] <-- {label}: status={row['status']}  "
              f"cruise_vehicle_OSPL={cn if cn is None else round(cn, 2)} dB  "
              f"({row.get('wall_s', 0):.0f} s)\n", flush=True)
        with open(os.path.join(_HERE, "reconcile_quiet_end.json"), "w") as f:
            json.dump(rows, f, indent=2, default=str)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'label':38s} {'cruise_single':>13s} {'cruise_vehicle':>15s} "
          f"{'hover_OSPL':>11s} {'cons_ok':>8s}  status", flush=True)
    for r in rows:
        cs = r.get("cruise_single_ospl_db")
        cn = r.get("cruise_vehicle_ospl_db")
        hv = r.get("hover_ospl_db")
        ok = all((r.get("constraint_checks") or {}).values()) if r["status"] == "ok" else None
        print(f"{r['label']:38s} "
              f"{('' if cs is None else format(cs, '.2f')):>13s} "
              f"{('' if cn is None else format(cn, '.2f')):>15s} "
              f"{('' if hv is None else format(hv, '.1f')):>11s} "
              f"{str(ok):>8s}  {r['status']}", flush=True)
    print(f"\njson -> {os.path.join(_HERE, 'reconcile_quiet_end.json')}", flush=True)


if __name__ == "__main__":
    main()
