from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import csdl_alpha as csdl
import numpy as np


ArrayLike = Union[np.ndarray, csdl.Variable]


@dataclass
class AcousticObserverData(csdl.VariableGroup):
    positions: ArrayLike
    frame: str = "rotor_local"
    velocities: Optional[ArrayLike] = None
    names: Optional[Sequence[str]] = None


@dataclass
class RotorAcousticSettings(csdl.VariableGroup):
    modes: Sequence[int] = (1, 2, 3)
    load_harmonics: Sequence[int] = tuple(range(11))
    tonal_model: str = "lowson"
    tonal_enabled: bool = False
    thickness_enabled: bool = False
    sears_enabled: bool = False
    sears_gust_amplification: float = 0.06
    broadband_enabled: bool = False
    broadband_model: str = "gill_lee"
    gill_lee_norm_hub_radius: float = 0.2
    broadband_center_frequencies: Sequence[float] = (
        12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0,
        125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,
        1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
        6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0, 25000.0,
        31500.0, 40000.0, 50000.0, 63000.0,
    )
    a_weighting_enabled: bool = True
    reference_pressure: float = 20.0e-6
    pressure_squared_floor: float = 4.0e-16


@dataclass
class RotorAcousticOutputs(csdl.VariableGroup):
    observer_distance: csdl.Variable
    observer_direction: csdl.Variable
    observer_axis_cosine: csdl.Variable
    blade_passing_frequencies: csdl.Variable
    tonal_pressure_squared: Optional[csdl.Variable] = None
    broadband_pressure_squared: Optional[csdl.Variable] = None
    total_pressure_squared: Optional[csdl.Variable] = None
    tonal_spl: Optional[csdl.Variable] = None
    loading_pressure_squared: Optional[csdl.Variable] = None
    loading_spl: Optional[csdl.Variable] = None
    loading_mode_spl: Optional[csdl.Variable] = None
    loading_cosine_pressure: Optional[csdl.Variable] = None
    loading_sine_pressure: Optional[csdl.Variable] = None
    thickness_pressure_squared: Optional[csdl.Variable] = None
    thickness_spl: Optional[csdl.Variable] = None
    thickness_mode_spl: Optional[csdl.Variable] = None
    thickness_cosine_pressure: Optional[csdl.Variable] = None
    thickness_sine_pressure: Optional[csdl.Variable] = None
    tonal_mode_spl: Optional[csdl.Variable] = None
    tonal_cosine_pressure: Optional[csdl.Variable] = None
    tonal_sine_pressure: Optional[csdl.Variable] = None
    broadband_spl: Optional[csdl.Variable] = None
    broadband_frequencies: Optional[csdl.Variable] = None
    broadband_one_third_octave_spl: Optional[csdl.Variable] = None
    broadband_one_third_octave_pressure_squared: Optional[csdl.Variable] = None
    total_spl: Optional[csdl.Variable] = None
    total_spl_a_weighted: Optional[csdl.Variable] = None
