Results folder.

Figs and text results of experiments saved here.

During training, real-time results can be displayed via [tensorboard](https://www.tensorflow.org/tensorboard):

```powershell
tensorboard --logdir=/results/log/tensorboard/<tb_log_exp_name>
```

Logs of hyper-parameter tuning runs are found in `/log` and can similarly be displayed ([RayTune docs](https://docs.ray.io/en/latest/tune/user-guide.html#tune-logging)).

## Reproducibility Instructions

To run all experiments from the paper:

- Main results

    ```posh
    python3 main.py --domain_shift hospital
    python3 main.py --domain_shift region
    python3 main.py --domain_shift ward
    python3 main.py --domain_shift ethnicity
    python3 main.py --domain_shift ethnicity_coarse
    python3 main.py --domain_shift age
    ```

- Alternative outcome definitions:

    ```posh
    python3 main.py --outcome Shock_4h
    python3 main.py --outcome Shock_12h
    python3 main.py --outcome ARF_4h
    python3 main.py --outcome ARF_12h
    ```

- Additional sequential models:

    ```posh
    python3 main.py --models RNN GRU LSTM
    ```

- Regularization experiments:

    ```posh
    python3 main.py --dropout
    ```

- Class incremental experiments:

    ```posh
    python3 main.py --class_shift
    ```
