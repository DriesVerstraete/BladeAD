from __future__ import annotations

from dataclasses import dataclass
import math
import csdl_alpha as csdl


@dataclass(frozen=True)
class McDonaldParameters:
    """Inputs to Shahjahan's implementation of the McDonald loss map."""

    peak_efficiency: float
    peak_efficiency_rpm: float
    peak_efficiency_torque: float
    k0: float
    c4: float
    c5: float
    c6: float
    c7: float
    c8: float
    c9: float
    c10: float
    c11: float
    c12: float
    efficiency_scale: float = 1.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(getattr(self, name)) for name in self.__dataclass_fields__):
            raise ValueError("McDonald parameters must be finite")
        if not 0.0 < self.peak_efficiency <= 1.0:
            raise ValueError("peak_efficiency must lie in (0, 1]")
        if self.peak_efficiency_rpm <= 0.0 or self.peak_efficiency_torque <= 0.0:
            raise ValueError("peak-efficiency speed and torque must be positive")
        if self.efficiency_scale <= 0.0:
            raise ValueError("efficiency_scale must be positive")

    @property
    def derived_coefficients(self) -> tuple[float, float, float, float]:
        eta = self.peak_efficiency
        omega = self.peak_efficiency_rpm * 2.0 * math.pi / 60.0
        torque = self.peak_efficiency_torque
        c0 = self.k0 * omega * torque / 6.0 * (1.0 - eta) / eta
        c1 = -3.0 * c0 / (2.0 * omega) + torque * (1.0 - eta) / (4.0 * eta)
        c2 = c0 / (2.0 * omega**3) + torque * (1.0 - eta) / (4.0 * eta * omega**2)
        c3 = omega * (1.0 - eta) / (2.0 * torque * eta)
        return c0, c1, c2, c3


@dataclass(frozen=True)
class ChebyshevTorqueEnvelope:
    """Differentiable continuous-torque curve fitted conservatively to motor data."""

    minimum_rpm: float
    maximum_rpm: float
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_rpm < self.maximum_rpm:
            raise ValueError("torque-envelope RPM limits are invalid")
        if not self.coefficients or not all(math.isfinite(value) for value in self.coefficients):
            raise ValueError("torque-envelope coefficients must be finite and non-empty")

    def evaluate(self, rpm: csdl.Variable) -> csdl.Variable:
        x = 2.0 * (rpm - self.minimum_rpm) / (self.maximum_rpm - self.minimum_rpm) - 1.0
        value = self.coefficients[0] + 0.0 * x
        if len(self.coefficients) == 1:
            return value
        previous = 1.0 + 0.0 * x
        current = x
        value = value + self.coefficients[1] * current
        for coefficient in self.coefficients[2:]:
            following = 2.0 * x * current - previous
            value = value + coefficient * following
            previous, current = current, following
        return value


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
    """Evaluate the full polynomial-and-logarithmic McDonald loss map."""

    c0, c1, c2, c3 = parameters.derived_coefficients
    fitted_loss = (
        c0 + c1 * angular_speed + c2 * angular_speed**3 + c3 * torque**2
        + parameters.c4 * torque
        + parameters.c5 * angular_speed * torque
        + parameters.c6 * angular_speed**2
        + parameters.c7 * angular_speed * torque**2
        + parameters.c8 * angular_speed**2 * torque
        + parameters.c9 * angular_speed * csdl.log(angular_speed)
        + parameters.c10 * torque * csdl.log(torque)
        + parameters.c11 * csdl.log(angular_speed)
        + parameters.c12 * csdl.log(torque)
    )
    shaft_power = angular_speed * torque
    efficiency = parameters.efficiency_scale * shaft_power / (shaft_power + fitted_loss)
    electrical_power = shaft_power / efficiency
    power_loss = electrical_power - shaft_power
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
