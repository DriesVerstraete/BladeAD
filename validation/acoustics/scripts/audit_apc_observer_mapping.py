import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "apc_11x4"
EXPERIMENT_TO_DRIVER_INDEX = {22.5: 3, 45.0: 4}


def downstream_angle_from_rotor_plane(position):
    x, y, _ = position
    if x < 0.0:
        raise ValueError("The broadband comparison requires a downstream observer.")
    return np.rad2deg(np.arctan2(x, abs(y)))


def audit_mapping():
    with (FIXTURE / "observers.csv").open(newline="") as stream:
        observers = list(csv.DictReader(stream))
    rows = []
    for experimental_angle, driver_index in EXPERIMENT_TO_DRIVER_INDEX.items():
        observer = observers[driver_index]
        position = np.array(
            [float(observer[name]) for name in ("x_m", "y_m", "z_m")]
        )
        derived_angle = downstream_angle_from_rotor_plane(position)
        rows.append(
            {
                "experimental_angle_from_rotor_plane_deg": experimental_angle,
                "driver_index": driver_index,
                "driver_angle_parameter_deg": float(observer["angle_deg"]),
                "position_m": position.tolist(),
                "derived_angle_from_rotor_plane_deg": derived_angle,
                "absolute_error_deg": abs(derived_angle - experimental_angle),
            }
        )
    return rows


def main():
    rows = audit_mapping()
    if any(row["absolute_error_deg"] > 1.0e-10 for row in rows):
        raise RuntimeError("APC observer mapping does not match the Cartesian geometry.")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
