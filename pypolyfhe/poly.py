from logging import getLogger
from enum import Enum, auto
import uuid

logger = getLogger(__name__)


class Poly:
    r"""Context-manager that changes the selected device.

    Args:
        device_idx (int): device index to select. Negative values are not allowed.
    """

    def __init__(self, degree: int, limb: int):
        if degree <= 0:
            logger.error("Degree must be a positive integer.")
            raise ValueError("Degree must be a positive integer.")
        if limb <= 0:
            logger.error("Limb must be a positive integer.")
            raise ValueError("Limb must be a positive integer.")
        self.degree = degree
        self.limb = limb


class PolyOpType(Enum):
    Add = auto()
    Accum = auto()
    Sub = auto()
    Mult = auto()
    MultConst = auto()
    MultKey = auto()
    MultKeyAccum = auto()
    Decomp = auto()
    BConv = auto()
    BConvGeneral = auto()
    ModDown = auto()
    ModUp = auto()
    NTT = auto()
    NTTPhase1 = auto()
    NTTPhase2 = auto()
    iNTT = auto()
    iNTTPhase1 = auto()
    iNTTPhase2 = auto()
    Init = auto()
    End = auto()
    # Special Edge
    Const = auto()
    InitEdge = auto()
    EndEdge = auto()

    def __str__(self):
        return self.name

    def __format__(self, format_spec):
        return self.name
