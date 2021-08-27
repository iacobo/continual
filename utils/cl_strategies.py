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

STRATEGIES = {'Naive':Naive, 'Joint':JointTraining, 'Cumulative':Cumulative,
'EWC':EWC, 'LwF':LwF, 'SI':SynapticIntelligence,
'Replay':Replay, 'GEM':GEM, 'AGEM':AGEM, 'GDumb':GDumb, 'CoPE':CoPE,
'AR1':AR1, 'StreamingLDA':StreamingLDA, 'CWRStar':CWRStar}
