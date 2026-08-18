# Hartzell F-9684-14 tonal-validation provenance

## Selected source

- J. W. Delfs, S. Proskurov, J. Thoma, and S. Langer, *A Propeller Noise Model for Aircraft
  Conceptual and Preliminary Design*, Research Square preprint, 2026,
  DOI `10.21203/rs.3.rs-9508453/v1`, CC BY 4.0.
- The preprint re-analyses the DLR/FAA DNW test campaign reported by W. Dobrzynski et al.,
  *DFVLR/FAA Propeller Noise Tests in the German-Dutch Wind Tunnel DNW*, DFVLR-IB 129-86/3,
  FAA-AEE-86-3, 1986.

The open preprint was chosen because it recovers the F-9684-14 geometry, atmospheric state,
measured thrust and power coefficients, spectral data, and an unambiguous observer: the reference
microphone in the propeller plane, 4 m from the axis. The source PDF is not committed; the compact
numerical transcription and its provenance are sufficient to reproduce this fixture.

## Case selection

BC-4 and AC-2 are the two square-tip cases nearest the BC-2 and AC-1 cases reused by Shahjahan
et al. They span circumferential tip Mach numbers 0.661 and 0.751. Shahjahan's AC-1/BC-2 Figure 5
was not frozen as the third validation fixture because the paper supplies only prose descriptions
for microphones 4 and 9, not complete coordinates; treating those curves as a common 4 m in-plane
observer would require an unsupported assumption.

## Extraction and modelling limits

- Table values are transcribed directly. RPM and dimensional thrust/power are reconstructed from
  the published nondimensional coefficients, diameter, density, and speed of sound.
- Figure 6 geometry is exact only at the labelled/table anchors. Inboard radial geometry and the
  first six Figure 7 experimental harmonics are manual plot digitizations, nominally to about
  1 dB in SPL.
- The paper states that the sections correspond essentially to Clark-Y profiles. BladeAD uses its
  existing Clark-Y polar; Reynolds-number variation is not represented.
- The runner scales BEM sectional thrust and drag independently to the measured `C_T` and `C_P`.
  The integrated aerodynamic source strengths are therefore exact by construction, while their
  radial distributions remain BEM/geometry dependent.
- The source reports no measured sectional loading and no conventional measurement-error bands.
  Its record-scatter metric was below 1.3 dB for the retained data.
- The source geometry is proprietary-test geometry reconstructed from a published plot. This is
  adequate for an independent validation diagnostic but not a manufacturing definition.
