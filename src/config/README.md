## config

Contains hyper-parameter configurations for models. 

Hyper-parameter search-space is specificed in `/config/config.py`. Default values tuned during paper experiments are defined in `/config/<dataset>/<outcome>/<domain>/config_<model>_<strategy>.json`.

## `main.py` arguments

Individual experiments can be specified with a combination of `--domain_shift` and `--outcome` parameters. A subset of models and Continual learning strategies can be evaluated with `--models` and `--strategies` respectively. To re-run hyperparameter tuning pass the `--validate` flag.

Example:

```posh
uv run main.py --domain_shift "hospital" --outcome "mortality_48h" --models "CNN" --strategies "EWC" "Replay"
```

Flag             | Arg(s)      | Meaning
-----------------|-------------|------------------------
`--domain_shift` | `region` `hospital` `age` `ethnicity`                      | Domain shift exhibited between tasks
`--outcome`      |`mortality_48h` `Shock_4h` `Shock_12h` `ARF_4h` `ARF_12h`   | Outcome to predict
`--models`       |`MLP` `CNN` `RNN` `LSTM` `GRU` `Transformer`                      | Model(s) to evaluate
`--strategies`   |`Naive` `Cumulative` `EWC` `OnlineEWC` `LwF` `SI` `GEM` `AGEM` `Replay` `GDumb` | Continual learning strategy(s) to evaluate
`--validate`     |                                                            | Re-tune hyper-parameters
`--num_samples`  |`<int>`                                                     | Budget for hyper-parameter search
