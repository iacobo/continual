# Continual Learning for Time-Series (and EHR)
Continual Learning repo for DPhil work

Project layout:

1.
    1. Adapt van de Ven [pytorch implementations](https://github.com/GMvandeVen/continual-learning) of main CL methods for ingesting time-series data as opposed to image  
       (i.e. convert CNN to RNN)
    3. Adapt proposed benchmarking datasets from [Harutyunyan et al](https://www.nature.com/articles/s41597-019-0103-9) (for *multi-task* learning on time-series EHR) into CL appropriate datasets.  
       (i.e. task incremental, where appropriate class/domain incremental by splitting on label/demographic)  
       (for DIL need to be careful of potential colinear variables acting as 'labels' for the domain e.g. splitting on "hospital" but "country of birth" is present as variable)
    5. Adapt 2 into format ingestible by 1. Evaluate 1. on 2.
2. Evaluate 1 on curated Oxford / Haven etc dataset. E.g. class-incremental learning predicting different health events / domain-incremental predicting generic health event from different specific events (respiratory failure, cardiac arrest etc).
3. Further development of superior techniques on more advanced dataset (domain-incremental learning with different datasets from oxford vs America vs SA etc).
4. *(Ambitious) pursue rigorous theoretically driven novel technique (as opposed to vague biological motivations / empirical architectures proposed).*
