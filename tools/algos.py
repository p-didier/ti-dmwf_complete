# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import numpy as np
from .base import Parameters
from dataclasses import dataclass

@dataclass
class Run:
    cfg: Parameters

    def launch(self):
        pass