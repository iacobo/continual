# Continual Learning for Time-Series (and EHR)
Continual Learning repo for DPhil work

Project layout:

1. Adapt van de Ven [pytorch implementations](https://github.com/GMvandeVen/continual-learning) of main CL methods for ingesting time-series data as opposed to image
    - i.e. convert CNN to RNN
2. Adapt proposed benchmarking time-series EHR multi-task learning datasets from [Harutyunyan et al](https://www.nature.com/articles/s41597-019-0103-9) into CL appropriate datasets.
3. Evaluate 1. on 2.
4. Evaluate 1 on our datasets. E.g. class-incremental learning predicting different health events / domain-incremental predicting generic health event from different specific events (respiratory failure, cardiac arrest etc).
