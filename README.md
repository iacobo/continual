# Continual Learning for Time-Series (and EHR)
Continual Learning repo for DPhil work

Project layout:

1.
    1. Adapt van de Ven [pytorch implementations](https://github.com/GMvandeVen/continual-learning) of main CL methods for ingesting time-series data as opposed to image (i.e. convert CNN to RNN)
    2. Adapt proposed benchmarking time-series EHR multi-task learning datasets from [Harutyunyan et al](https://www.nature.com/articles/s41597-019-0103-9) into CL appropriate datasets.
    3. Adapt 2 into format ingestible by 1. Evaluate 1. on 2.
2. Evaluate 1 on curated Oxford / Haven etc dataset. E.g. class-incremental learning predicting different health events / domain-incremental predicting generic health event from different specific events (respiratory failure, cardiac arrest etc).
3. Further development of superior techniques on more advanced dataset (domain-incremental learning with different datasets from oxford vs America vs SA etc).
4. *(Ambitious) pursue rigorous theoretically driven novel technique (as opposed to vague biological motivations / empirical architectures proposed).*
