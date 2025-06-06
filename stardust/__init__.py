"""init file for stardust package"""


# miscellaneous imports
from ._misc import vbprint, plot_circle, plot_sphere_wireframe

# pre-implemented dynamics
from .dynamics import *

# two-stage shooting-based optimization problems
from ._twostage_base import _BaseTwoStageOptimizer
from ._twostage_fixedtime_lsq import FixedTimeTwoStageLeastSquares
from ._twostage_fixedtime_min import FixedTimeTwoStageMinimizer
from ._twostage_fixedtime_primervector import FixedTimeTwoStagePrimerVector
try:
    from ._twostage_fixedtime_udp import FixedTimeTwoStageUDP
except:
    print("WARNING: pygmo not found, skipping FixedTimeTwoStageUDP")
    pass

# one-stage NLP
from ._onestage_nlp_fixedtime import FixedTimeShootingNLP

# scp
from ._scp_fixedtime import FixedTimeSCP