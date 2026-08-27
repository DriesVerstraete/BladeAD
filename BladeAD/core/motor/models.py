from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import csdl_alpha as csdl


@dataclass(frozen=True)
class McDonaldParameters:
    """Coefficients for ``sum(C_ij * torque**i * angular_speed**j)`` in watts."""

    loss_coefficients: Mapping[tuple[int, int], float]

    def __post_init__(self) -> None:
        if not self.loss_coefficients:
            raise ValueError("loss_coefficients must not be empty")
        for powers, coefficient in self.loss_coefficients.items():
            if (
                not isinstance(powers, tuple)
                or len(powers) != 2
                or any(not isinstance(power, int) or power < 0 for power in powers)
            ):
                raise ValueError("coefficient keys must be non-negative integer (torque, speed) powers")
            if not math.isfinite(coefficient) or coefficient < 0.0:
                raise ValueError("McDonald loss coefficients must be finite and non-negative")


@dataclass(frozen=True)
class ThreeConstantParameters:
    """Equivalent-circuit constants using rad/s/V, ohm, and ampere units."""

    speed_constant: float
    resistance: float
    no_load_current: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.speed_constant) or self.speed_constant <= 0.0:
            raise ValueError("speed_constant must be finite and positive in rad/s/V")
        if not math.isfinite(self.resistance) or self.resistance <= 0.0:
            raise ValueError("resistance must be finite and positive in ohm")
        if not math.isfinite(self.no_load_current) or self.no_load_current < 0.0:
            raise ValueError("no_load_current must be finite and non-negative in A")


@dataclass
class MotorOutputs(csdl.VariableGroup):
    """AD-connected motor outputs; current and voltage are model-dependent."""

    shaft_power: csdl.Variable
    electrical_power: csdl.Variable
    power_loss: csdl.Variable
    efficiency: csdl.Variable
    current: csdl.Variable | None = None
    voltage: csdl.Variable | None = None


def evaluate_mcdonald(
    angular_speed: csdl.Variable,
    torque: csdl.Variable,
    parameters: McDonaldParameters,
) -> MotorOutputs:
    """Evaluate McDonald's positive-polynomial loss model."""

    power_loss = None
    for (torque_power, speed_power), coefficient in parameters.loss_coefficients.items():
        term = coefficient * torque**torque_power * angular_speed**speed_power
        power_loss = term if power_loss is None else power_loss + term

    shaft_power = angular_speed * torque
    electrical_power = shaft_power + power_loss
    efficiency = shaft_power / electrical_power
    return MotorOutputs(
        shaft_power=shaft_power,
        electrical_power=electrical_power,
        power_loss=power_loss,
        efficiency=efficiency,
    )


def evaluate_three_constant(
    angular_speed: csdl.Variable,
    torque: csdl.Variable,
    parameters: ThreeConstantParameters,
) -> MotorOutputs:
    """Evaluate the Kv--resistance--no-load-current equivalent circuit."""

    useful_voltage = angular_speed / parameters.speed_constant
    current = parameters.no_load_current + parameters.speed_constant * torque
    voltage = useful_voltage + current * parameters.resistance
    shaft_power = angular_speed * torque
    electrical_power = current * voltage
    power_loss = electrical_power - shaft_power
    efficiency = shaft_power / electrical_power
    return MotorOutputs(
        shaft_power=shaft_power,
        electrical_power=electrical_power,
        power_loss=power_loss,
        efficiency=efficiency,
        current=current,
        voltage=voltage,
    )


def evaluate_motor(
    model: str,
    angular_speed: csdl.Variable,
    torque: csdl.Variable,
    parameters: McDonaldParameters | ThreeConstantParameters,
) -> MotorOutputs:
    """Evaluate the selected motor model with a common output interface."""

    if model == "mcdonald":
        if not isinstance(parameters, McDonaldParameters):
            raise TypeError("mcdonald requires McDonaldParameters")
        return evaluate_mcdonald(angular_speed, torque, parameters)
    if model == "three_constant":
        if not isinstance(parameters, ThreeConstantParameters):
            raise TypeError("three_constant requires ThreeConstantParameters")
        return evaluate_three_constant(angular_speed, torque, parameters)
    raise ValueError("model must be 'mcdonald' or 'three_constant'")
