from .models import ChebyshevTorqueEnvelope, McDonaldParameters


SHAHJAHAN_EMRAX188_PARAMETERS = McDonaldParameters(
    peak_efficiency=9.26684101e-1,
    peak_efficiency_rpm=5.62802036e3 / 1.89,
    peak_efficiency_torque=1.64415051e2 * 1.89,
    k0=-2.21815062e-4,
    c4=8.76596794e-4,
    c5=-8.69972007e-2,
    c6=-6.77381770e-3,
    c7=8.08260013e-4,
    c8=8.00176845e-5,
    c9=-6.91650617e-3,
    c10=7.62639412e-3,
    c11=-2.09303985e-2,
    c12=6.83102582e-2,
    efficiency_scale=0.962 / 0.91651294631772,
)


SHAHJAHAN_EMRAX188_CONTINUOUS_TORQUE = ChebyshevTorqueEnvelope(
    minimum_rpm=0.0,
    maximum_rpm=6500.0 / 1.89,
    coefficients=(
        90.2849670960357,
        -2.2923597488965624,
        -8.081869745200919,
        -0.6029350481069101,
        -0.3856511125427771,
        -0.06167212157675278,
        0.2074092597707926,
        -0.006701382490419908,
        0.0992730187156856,
        0.008319230872207144,
        0.002002339808171719,
    ),
)
