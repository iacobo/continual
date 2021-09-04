## config

Contains hyper-parameter configurations for models. 

Hyper-parameter search-space is specificed in `/config/config.py`. Default values tuned during paper experiments are defined in `/config/best_config_<dataset>_<domain>_<outcome>.json`.

## `main.py` arguments

Individual experiments can be specified with a combination of `--experiment` `--outcome` parameters. A subset of models and Continual learning strategies can be evaluated with `--models` and `--strategies` respectively.

To re-run hyperparameter tuning pass the `--validate` flag.

Flag           | Arg(s)      | Meaning
---------------|-------------|------------------------
`--experiment` | `region hospital age ethnicity` | Domain increment to use
`--outcome`    |`mortality_48h shock_4h shock_12h ARF_4h ARF_12h`       | Outcome to predict
`--models`     |`MLP CNN RNN LSTM Transformer`   | Model(s) to evaluate
`--strategies` |`Naive Cumulative EWC LwF SI GEM AGEM Replay GDumb` | Continual learning strategy(s) to evaluate
`--validate`   |             | Rerun hyper-parameter search
`--num_samples` |`<int>`         | Budget for hyper-parameter search