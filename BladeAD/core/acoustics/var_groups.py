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
    tonal_enabled: bool = False
    thickness_enabled: bool = False
    broadband_enabled: bool = False
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
    thickness_pressure_squared: Optional[csdl.Variable] = None
    thickness_spl: Optional[csdl.Variable] = None
    thickness_mode_spl: Optional[csdl.Variable] = None
    tonal_mode_spl: Optional[csdl.Variable] = None
    tonal_cosine_pressure: Optional[csdl.Variable] = None
    tonal_sine_pressure: Optional[csdl.Variable] = None
    broadband_spl: Optional[csdl.Variable] = None
    total_spl: Optional[csdl.Variable] = None
    total_spl_a_weighted: Optional[csdl.Variable] = None
