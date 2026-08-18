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
- The raw chord and twist stations below were previously transcribed into the SPL project's
  `validate_dji9443_hover.py` and independently exercised at 5400 RPM.

## Current fixture boundary

- Rotor radius, hub radius, blade count, chord, twist, RPM, density, and measured
  `C_T = 0.072` are frozen here.
- Experimental narrowband spectrum, BPF1/BPF2 directivity, OASPL directivity, microphone
  coordinates, acoustic normalization, and measurement uncertainty are **not yet transcribed**.
- This is therefore an aerodynamic-source fixture and provenance anchor, not yet an executable
  acoustic validation fixture. No BladeAD tonal-accuracy claim may use it until those missing
  primary-source quantities are digitized and independently checked.
- FLOWUnsteady's current documentation warns that the original 2020 broadband comparison used
  the wrong microphone plane. Only the corrected current tutorial data may be accepted.
