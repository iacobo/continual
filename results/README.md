Results folder.

Figs and text results of experiments saved here. 

During training, real-time results can be displayed via [tensorboard](https://www.tensorflow.org/tensorboard):

```powershell
tensorboard --logdir=/results/log/tensorboard/<tb_log_exp_name>
```

Logs of hyper-parameter tuning runs are found in `/log` and can similarly be displayed ([RayTune docs](https://docs.ray.io/en/latest/tune/user-guide.html#tune-logging)).