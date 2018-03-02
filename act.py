# from enum import IntEnum
from enum import Enum


class PlayerAct(Enum):
    up = 0
    right = 1
    down = 2
    left = 3


class LiarAct(Enum):
    up = 0
    right = 1
    down = 2
    left = 3

    @staticmethod
    def toVec(a):
        vec = [0 for i in range(4)]
        vec[a] = 1
        return vec
