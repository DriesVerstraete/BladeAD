# DJI 9443 hover tonal validation provenance

## Primary experimental source

- N. S. Zawodny, D. D. Boyd Jr., and C. L. Burley, “Acoustic Characterization and
  Prediction of Representative, Small-Scale Rotary-Wing Unmanned Aircraft System
  Components,” VFS 72nd Annual Forum, 2016, DOI `10.4050/F-0072-2016-11346`.
- NASA NTRS document `20160009054`, public-use US Government work.
- The experiment reports hover performance and acoustics for two isolated two-bladed,
  fixed-pitch rotors across multiple rotation rates. It explicitly identifies motor noise as a
  material source in addition to rotor noise.

## Geometry source

- FLOWUnsteady's public MIT-licensed DJI 9443 rotor database, ultimately sourced from the
  Zawodny et al. rotor geometry.
- Source repository commit: `b7283db2e94a5f44a7ef2d57f223b0bdb8d0dec7`.
- The raw chord and twist stations below were previously transcribed into the SPL project's
  `validate_dji9443_hover.py` and independently exercised at 5400 RPM.
- The seven source polar tables and their radial mapping are now frozen under `airfoil_polars/`
  and `airfoil_sections.csv`. They are copied from FLOWUnsteady without numerical alteration.
- FLOWUnsteady treats the mapping coordinate as normalized span from hub to tip and converts it
  with `(Rhub + position*(Rtip-Rhub))/Rtip`; the normalized fixture column records this executable
  convention rather than preserving the upstream CSV's misleading `r/R` label.
- Polar SHA-256 values, in section order, are:
  `831fadfdf98315d07b47f2f70077ab7c64ae60c4b0723023a2e463aa38adec97`,
  `8a75dc4a31cd9daf0a16925e89c8bd40da41b55851f507263952fa524a9b3d8b`,
  `5cc924814c04302cfabbf1b539b51a6b7aad61b6a4d55a31ee3d41befca96a62`,
  `ad6f314a308a85939dad81ba77a63d467aa34d1cca6c95382911e28b9341a273`,
  `624b38db8cb2b7af25dc4f8c89c74c00ea038a9c3a76552bb4f754bfed72ce1d`,
  `7b2391b57f5ddd560fca732c8e514f09fcb40780e21a632279ce18e729a7f3ca`, and
  `57ed68f0b2a9252fe0257aad5d95ed7e4b30ad08d8114f379c308cb58d9ca74a`.
- FLOWUnsteady distributes these files under its MIT licence. The upstream licence and copyright
  notice remain authoritative; this fixture records provenance rather than changing ownership.

## Acoustic transcription

- The experimental BPF1/BPF2 and OASPL points are transcribed from the current FLOWUnsteady
  digitizations of Zawodny et al. Figures 14 and 12, respectively.
- Upstream file SHA-256 values are `97b6e2c73aa4c66a93d021c60afdcccbffeb4548cd6ce30ceac78d803ced8ccd`
  (BPF1), `d3984a6cb8975e51438cab5f939d72a2aaf31cf65952f6c153c8e91394abe3d2`
  (BPF2), and `dac1be65126b84ad4b19b58b761a6960ad303682b1bb90edf90c6e13334a72ea`
  (OASPL).
- The -45-degree narrowband and one-third-octave source hashes are
  `caa410d6c622cb1f2058ed4452dbfca04e698b41164499c20321e505df86ffc5` and
  `6deea070044d5eb0c34773c21e0141eb8ac9e70af4b18097d55f77b0828e8000`.
- The current FLOWUnsteady tutorial uses a 1.905 m circular microphone array, 5400 RPM,
  density 1.071778 kg/m3, dynamic viscosity 1.85508e-5 Pa s, and sound speed 342.35 m/s.
- Observer angles are measured from the rotor plane, with +90 degrees upstream. The runner maps
  upstream onto BladeAD's positive thrust-vector axis; this reverses FLOWUnsteady's plotted
  x-axis orientation without changing the reported observer angles.
- FLOWUnsteady warns that an older broadband comparison used the wrong microphone plane. These
  fixture files use only the corrected current tutorial data.

## Interpretation boundary

- Rotor radius, hub radius, blade count, chord, twist, RPM, atmosphere, observer geometry, and
  measured `C_T = 0.072` are frozen here.
- The published directivity points are digitized rather than tabulated; digitization uncertainty
  and measurement uncertainty are not reported in the recovered machine-readable sources.
- OASPL includes broadband rotor and motor noise and must not be used as the acceptance target
  for loading-only tonal models. BPF1/BPF2 are the tonal comparison quantities.
- The current FLOWUnsteady tutorial uses the same narrowband digitization in A-weighted and
  unweighted plotting sections. Its weighting state is therefore unresolved and it is retained
  for diagnostic plotting rather than quantitative acceptance.
- The private local paper is supporting evidence only and remains excluded from Git.
