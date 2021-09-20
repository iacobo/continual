"""
Continual learning strategies.

Contains:

- Baselines
- Regularization methods
- Rehearsal methods
- Misc
"""
from avalanche.training.strategies import Naive, JointTraining, Cumulative # Baselines
from avalanche.training.strategies import EWC, LwF, SynapticIntelligence   # Regularisation
from avalanche.training.strategies import Replay, GDumb, GEM, AGEM         # Rehearsal
from avalanche.training.strategies import AR1, CWRStar, CoPE, StreamingLDA

STRATEGIES = {
    # Baselines
    'Naive':Naive, 'Naive_no_reg':Naive, 'Joint':JointTraining, 'Cumulative':Cumulative,
    # Regularization based
    'EWC':EWC, 'OnlineEWC':EWC, 'LwF':LwF, 'SI':SynapticIntelligence, #'LFL':LFL, 
    # Replay
    'Replay':Replay, 'GEM':GEM, 'AGEM':AGEM, 'GDumb':GDumb, 'CoPE':CoPE,
    # Misc.
    'AR1':AR1, 'StreamingLDA':StreamingLDA, 'CWRStar':CWRStar}
