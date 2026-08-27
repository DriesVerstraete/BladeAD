from .models import (
    ChebyshevTorqueEnvelope,
    McDonaldParameters,
    MotorOutputs,
    ThreeConstantParameters,
    evaluate_mcdonald,
    evaluate_motor,
    evaluate_three_constant,
)
from .presets import SHAHJAHAN_EMRAX188_CONTINUOUS_TORQUE, SHAHJAHAN_EMRAX188_PARAMETERS

__all__ = [
    "ChebyshevTorqueEnvelope",
    "McDonaldParameters",
    "MotorOutputs",
    "ThreeConstantParameters",
    "evaluate_mcdonald",
    "evaluate_motor",
    "evaluate_three_constant",
    "SHAHJAHAN_EMRAX188_CONTINUOUS_TORQUE",
    "SHAHJAHAN_EMRAX188_PARAMETERS",
]
