from __future__ import annotations

import csdl_alpha as csdl


def a_weighting_db(frequency_hz):
    frequency_squared = frequency_hz**2
    numerator = 12194.0**2 * frequency_squared**2
    denominator = (
        (frequency_squared + 20.6**2)
        * csdl.sqrt(
            (frequency_squared + 107.7**2)
            * (frequency_squared + 737.9**2)
        )
        * (frequency_squared + 12194.0**2)
    )
    return 20.0 / csdl.log(10.0) * csdl.log(numerator / denominator) + 2.0
