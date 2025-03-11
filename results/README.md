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
    uv run main.py --domain_shift "hospital"
    uv run main.py --domain_shift "region"
    uv run main.py --domain_shift "ward"
    uv run main.py --domain_shift "ethnicity"
    uv run main.py --domain_shift "ethnicity_coarse"
    uv run main.py --domain_shift "age"
    ```

- Alternative outcome definitions:

    ```posh
    uv run main.py --outcome "Shock_4h"
    uv run main.py --outcome "Shock_12h"
    uv run main.py --outcome "ARF_4h"
    uv run main.py --outcome "ARF_12h"
    ```

- Additional sequential models:

    ```posh
    uv run main.py --models "RNN" "GRU" "LSTM"
    ```

- Regularization experiments:

    ```posh
    uv run main.py --dropout
    ```

- Class incremental experiments:

    ```posh
    uv run main.py --class_shift
    ```
