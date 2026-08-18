from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten_serializable(prefix, value, output, seen, depth=0):
    if depth > 16 or id(value) in seen:
        return
    if isinstance(value, Mapping):
        seen.add(id(value))
        for key, child in value.items():
            flatten_serializable(f"{prefix}.{key}", child, output, seen, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        array = np.asarray(value)
        if array.dtype != object:
            output[prefix] = array.copy()
            return
        seen.add(id(value))
        for index, child in enumerate(value):
            flatten_serializable(f"{prefix}.{index}", child, output, seen, depth + 1)
        return
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            output[prefix] = value.copy()
        return
    if isinstance(value, (str, bytes, bool, int, float, complex, np.generic)):
        output[prefix] = np.asarray(value)


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcaide-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures",
    )
    args = parser.parse_args()
    rcaide_root = args.rcaide_root.resolve()
    sys.path.insert(0, str(rcaide_root))
    numpy_trapezoid_compatibility_alias = not hasattr(np, "trapezoid")
    if numpy_trapezoid_compatibility_alias:
        np.trapezoid = np.trapz

    validation = load_module(
        "frequency_domain_baseline_source",
        rcaide_root
        / "VnV"
        / "Verification"
        / "analysis_aeroacoustics"
        / "frequency_domain_test.py",
    )
    original_compute = validation.compute_rotor_noise
    captures = []

    def capture_compute(microphones, rotor, segment, settings):
        original_compute(microphones, rotor, segment, settings)
        arrays = {"observer.position_m": np.asarray(microphones).copy()}
        conditions = segment.state.conditions
        flatten_serializable("acoustics", conditions.aeroacoustics, arrays, set())
        flatten_serializable("aerodynamics", conditions.aerodynamics, arrays, set())
        flatten_serializable("energy", conditions.energy, arrays, set())
        flatten_serializable("freestream", conditions.freestream, arrays, set())
        flatten_serializable("frames", conditions.frames, arrays, set())
        flatten_serializable("rotor", rotor, arrays, set())
        flatten_serializable("settings", settings, arrays, set())
        captures.append({"tag": rotor.tag, "fidelity": settings.fidelity, "arrays": arrays})

    validation.compute_rotor_noise = capture_compute
    plot_parameters = validation.plot_parameters()
    validation.Harmonic_Noise_Validation(plot_parameters)
    validation.Broadband_Noise_Validation(plot_parameters)

    for capture in captures:
        if capture["tag"] == "F8745_D4_Propeller":
            case_dir = args.output_root / "f8745_d4"
            filename = f"rcaide_{capture['fidelity']}_baseline.npz"
        elif capture["tag"] == "APC_11x4_Propeller":
            case_dir = args.output_root / "apc_11x4"
            filename = f"rcaide_{capture['fidelity']}_baseline.npz"
        else:
            raise ValueError(f"Unexpected rotor tag {capture['tag']}")
        arrays = capture["arrays"]
        arrays["source_commit"] = np.asarray(args.source_commit)
        arrays["source_driver"] = np.asarray(
            "VnV/Verification/analysis_aeroacoustics/frequency_domain_test.py"
        )
        driver_path = (
            rcaide_root
            / "VnV"
            / "Verification"
            / "analysis_aeroacoustics"
            / "frequency_domain_test.py"
        )
        arrays["source_driver_sha256"] = np.asarray(file_sha256(driver_path))
        arrays["runtime_python"] = np.asarray(sys.version)
        arrays["runtime_platform"] = np.asarray(platform.platform())
        arrays["runtime_numpy"] = np.asarray(np.__version__)
        arrays["runtime_numpy_trapezoid_compatibility_alias"] = np.asarray(
            numpy_trapezoid_compatibility_alias
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        output_path = case_dir / filename
        np.savez_compressed(output_path, **arrays)
        manifest = {
            "file": filename,
            "rotor_tag": capture["tag"],
            "fidelity": capture["fidelity"],
            "source_commit": args.source_commit,
            "source_driver_sha256": file_sha256(driver_path),
            "numpy_trapezoid_compatibility_alias": numpy_trapezoid_compatibility_alias,
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(arrays.items())
            },
        }
        manifest_path = output_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
