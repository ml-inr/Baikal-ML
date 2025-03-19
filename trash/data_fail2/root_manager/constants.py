import numpy as np
from dataclasses import dataclass

@dataclass
class Constants:
    # Water properties
    N: float = 1.37
    COS_C: float = 1 / N
    SIN_C: float = np.sqrt(1 - COS_C**2)
    TAN_C: float = SIN_C / COS_C
    C_PART: float = 299792458.0  # Speed of particles in m/s
    C_LIGHT: float = 218826621.0  # Speed of light in water in m/s
    EPSILON: float = 1e-9  # Small value to avoid division by zero

    # Fixed values used in channels and other calculations
    CHANNEL_DIVISOR: int = 288
    STRING_DIVISOR: int = 36