from __future__ import annotations

import csv

import numpy as np
from scipy.special import jv

from run_bladead_f8745_validation import FIXTURE, REPORTS, evaluate_f8745


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def reproduce_rcaide_terms():
    observers = np.genfromtxt(FIXTURE / "observers.csv", delimiter=",", names=True)
    with np.load(FIXTURE / "rcaide_line_source_baseline.npz", allow_pickle=False) as archive:
        radius = archive["rotor.radius_distribution"]
        tip_radius = float(archive["rotor.tip_radius"])
        chord = archive["rotor.chord_distribution"]
        thickness_to_chord = archive["rotor.thickness_to_chord"]
        y_upper = archive["rotor.airfoils.airfoil.geometry.y_upper_surface"]
        y_lower = archive["rotor.airfoils.airfoil.geometry.y_lower_surface"]
        thrust = archive[
            "energy.converters.F8745_D4_Propeller.disc_thrust_distribution"
        ]
        torque = archive[
            "energy.converters.F8745_D4_Propeller.disc_torque_distribution"
        ]
        omega = archive["energy.converters.F8745_D4_Propeller.omega"][:, 0]
        density = archive["freestream.density"][:, 0]
        sound_speed = archive["freestream.speed_of_sound"][:, 0]
        velocity = archive["frames.inertial.velocity_vector"]
        archived_spl = archive[
            "acoustics.converters.F8745_D4_Propeller.SPL_harmonic_bpf_spectrum"
        ][:, (6, 9), 1:19]
        wing_wake_adjustment = float(archive["settings.wing_wake_interactional_dB_adjustment"])

    blade_count = 2
    modes = np.arange(1, 19)
    load_harmonics = np.arange(6)
    z = radius / tip_radius
    x_edge = np.linspace(-0.5, 0.5, len(y_upper) + 1)
    x = 0.5 * (x_edge[:-1] + x_edge[1:])
    full_thickness = y_upper - y_lower
    loading = np.zeros((3, 2, len(modes)), dtype=complex)
    thickness = np.zeros_like(loading)
    for case in range(3):
        mach = np.linalg.norm(velocity[case]) / sound_speed[case]
        axial_mach = velocity[case, 0] / sound_speed[case]
        tip_mach = tip_radius * omega[case] / sound_speed[case]
        axial_fft = np.fft.rfft(thrust[case] / tip_radius, axis=1)
        circumferential_fft = np.fft.rfft(
            torque[case] / (tip_radius * radius[:, None]), axis=1
        )
        for observer, observer_index in enumerate((6, 9)):
            position = np.array(
                [
                    observers["x_m"][observer_index],
                    observers["y_m"][observer_index],
                    observers["z_m"][observer_index],
                ]
            )
            geometric_distance = np.linalg.norm(position)
            geometric_angle = np.arccos(position[0] / geometric_distance)
            sine = np.sin(geometric_angle)
            retarded_angle = np.arccos(
                np.cos(geometric_angle) * np.sqrt(1.0 - mach**2 * sine**2)
                + mach * sine**2
            )
            in_plane_distance = np.linalg.norm(position[1:])
            retarded_distance = in_plane_distance / np.sin(retarded_angle)
            azimuth = np.arctan2(position[1], position[2])
            phi_prime = np.arccos(np.cos(azimuth))
            convection = 1.0 - mach * np.cos(retarded_angle)
            helicoid = np.arctan(axial_mach / (z * tip_mach))
            relative_mach_squared = mach**2 + (z * tip_mach) ** 2
            for mode_index, mode in enumerate(modes):
                n = mode * blade_count
                propagation = np.exp(
                    1j
                    * n
                    * omega[case]
                    / sound_speed[case]
                    * retarded_distance
                    / convection
                )
                loading_sum = 0.0j
                for harmonic in load_harmonics:
                    argument = n * z * tip_mach * np.sin(retarded_angle) / convection
                    bessel = jv(n - harmonic, argument)
                    axial_term = (
                        n * z * tip_mach * np.cos(retarded_angle) / convection
                    ) * axial_fft[:, harmonic]
                    circumferential_term = -(
                        n - harmonic
                    ) * circumferential_fft[:, harmonic]
                    integrand = (axial_term + circumferential_term) * bessel / z
                    loading_sum += np.trapz(integrand, x=z) * np.exp(
                        1j * (n - harmonic) * (phi_prime - np.pi / 2.0)
                    )
                loading[case, observer, mode_index] = (
                    1j
                    * blade_count
                    * propagation
                    * loading_sum
                    / (4.0 * np.pi * retarded_distance * convection)
                )

                chordwise_wavenumber = chord / tip_radius * (
                    n * np.cos(helicoid) / z
                    + n
                    * tip_mach
                    * np.cos(retarded_angle)
                    * np.sin(helicoid)
                    / convection
                )
                thickness_shape = full_thickness[None, :] / chord[:, None]
                shape_function = np.trapz(
                    thickness_shape
                    * np.exp(1j * chordwise_wavenumber[:, None] * x[None, :]),
                    x=x,
                    axis=1,
                )
                bessel = jv(
                    n, n * z * tip_mach * np.sin(retarded_angle) / convection
                )
                thickness_integrand = (
                    relative_mach_squared
                    * chordwise_wavenumber**2
                    * thickness_to_chord
                    * shape_function
                    * bessel
                )
                thickness_sum = np.trapz(thickness_integrand, x=z) * np.exp(
                    1j * n * (phi_prime - np.pi / 2.0)
                )
                thickness[case, observer, mode_index] = (
                    -density[case]
                    * sound_speed[case] ** 2
                    * blade_count
                    * propagation
                    * thickness_sum
                    / (
                        4.0
                        * np.pi
                        * (retarded_distance / tip_radius)
                        * convection
                    )
                )
    reproduced_spl = 20.0 * np.log10(
        (np.abs(loading) + np.abs(thickness)) / 20.0e-6
    ) + wing_wake_adjustment
    return loading, thickness, reproduced_spl, archived_spl, wing_wake_adjustment


def main():
    loading, thickness, reproduced, archived, wing_wake_adjustment = reproduce_rcaide_terms()
    rows = []
    for case in range(3):
        for observer, angle in enumerate((60, 90)):
            difference = reproduced[case, observer] - archived[case, observer]
            rows.append(
                {
                    "case": f"F8745-D4-{case + 1}",
                    "reported_observer_angle_deg": angle,
                    "loading_peak_overall_db": 20.0
                    * np.log10(np.linalg.norm(np.abs(loading[case, observer])) / 20.0e-6),
                    "thickness_peak_overall_db": 20.0
                    * np.log10(np.linalg.norm(np.abs(thickness[case, observer])) / 20.0e-6),
                    "maximum_archive_reproduction_error_db": np.max(np.abs(difference)),
                    "mean_archive_reproduction_error_db": np.mean(difference),
                }
            )
    _write_csv(REPORTS / "f8745_hanson_rcaide_term_audit.csv", rows)
    bladead = evaluate_f8745(tonal_model="hanson_line", return_components=True)
    loading_pressure = (
        bladead["loading_cosine_pressure"]
        + 1j * bladead["loading_sine_pressure"]
    )
    thickness_pressure = (
        bladead["thickness_cosine_pressure"]
        + 1j * bladead["thickness_sine_pressure"]
    )
    coherent_peak = 20.0 * np.log10(
        np.abs(loading_pressure + thickness_pressure) / 20.0e-6
    )
    magnitude_peak = 20.0 * np.log10(
        (np.abs(loading_pressure) + np.abs(thickness_pressure)) / 20.0e-6
    )
    rcaide_unadjusted = archived - wing_wake_adjustment
    convention_rows = []
    variants = {
        "coherent_rms_production": bladead["combined"],
        "coherent_peak": coherent_peak,
        "magnitude_sum_peak": magnitude_peak,
    }
    for name, values in variants.items():
        for case in range(3):
            for observer, angle in enumerate((60, 90)):
                difference = values[case, observer] - rcaide_unadjusted[case, observer]
                convention_rows.append(
                    {
                        "convention": name,
                        "case": f"F8745-D4-{case + 1}",
                        "reported_observer_angle_deg": angle,
                        "mean_signed_difference_db": np.mean(difference),
                        "mean_absolute_difference_db": np.mean(np.abs(difference)),
                        "maximum_absolute_difference_db": np.max(np.abs(difference)),
                    }
                )
    _write_csv(REPORTS / "f8745_hanson_pressure_convention_audit.csv", convention_rows)
    lines = [
        "# F8745 RCAIDE Hanson term audit",
        "",
        "This script independently reproduces the archived RCAIDE line-source equation sequence",
        "from frozen arrays, including retarded geometry, raw azimuth FFT, element-force radial",
        "integration, thickness normalization, and magnitude-only peak-pressure summation.",
        f"The archived configuration then adds a uniform {wing_wake_adjustment:.1f} dB wing-wake",
        "interaction adjustment to every harmonic. This adjustment is reproduced here only for",
        "audit parity and is not part of BladeAD's production Hanson physics.",
        "",
        "| Case | Angle | Loading peak overall | Thickness peak overall | Max archive error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['reported_observer_angle_deg']} | "
            f"{row['loading_peak_overall_db']:.3f} | "
            f"{row['thickness_peak_overall_db']:.3f} | "
            f"{row['maximum_archive_reproduction_error_db']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## BladeAD pressure-convention isolation",
            "",
            "All rows use identical BladeAD component pressures and compare against RCAIDE with",
            "the 15 dB adjustment removed.",
            "",
            "| Convention | Mean absolute difference range (dB) |",
            "|---|---:|",
        ]
    )
    for name in variants:
        selected = [row for row in convention_rows if row["convention"] == name]
        values = [row["mean_absolute_difference_db"] for row in selected]
        lines.append(f"| {name} | {min(values):.3f}–{max(values):.3f} |")
    (REPORTS / "f8745_hanson_rcaide_term_audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
