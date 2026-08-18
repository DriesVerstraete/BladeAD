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
- Contour SHA-256 values, for sections 1 through 6, are
  `24403cb37ac10113eb76e5e81576cd06bf5af0ff1b29892cfa2f19307c0bd0a8`,
  `d18d7f5dff26248de85402eb0a2e68aaaab8fa74b5cfc0d9d10b593a57fe54bc`,
  `40320e04b10349f64eb8ee770a7170b61ed3fd57a404be1cb7b4df8f16e39fa7`,
  `1ac19a5df37906409cf7e7749d694e764ec13514874f8922683b009ad44ee402`,
  `d546470e8cede2d90e272971019c9ea3d891456ee20a86332ca79ed7d853b3af`, and
  `74df37decf127a1d90b7233d387f013e2a5c87fb6c12bdf0f0aafd09b9ed28b6`.
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
- The Figure 14 BPF2 computational references retain separate OF2-PSW/PAS and
  thickness/loading/total labels. Their upstream SHA-256 values are
  `84c498feb4ba51a45e7732d6b2bfa86dc447afb79c1733fa747fc71b1cff3f49`,
  `aa40da6310d4aa2394e0f29afe1a8f0b94690d58734ab20ba13fa215ed2d5eb5`,
  `1b46929a882187a89dae6dbd0ca390185d1ec36b845e1422f01117b789822ea9`,
  `c408bf19a2a00ad6aadbaa3ea50ef3e2842c41922354cd1fb4358bd5a6a7236b`,
  `84617091c4e347bcd2750ff3ca918fe03405d6c82bce6015a338135e31ae2bff`, and
  `c2275b9d93171d9c92a10ea225efff83b0d0b367cd9cec91470cdaa65ca38b50` in
  thickness-OF2, thickness-PAS, loading-OF2, loading-PAS, total-OF2, total-PAS order.
