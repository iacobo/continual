# Continual Learning of Longitudinal Health Records / ICU EHR

Repo for reproducing the experiments in *Continual Learning of Longitudinal Health Records* (arxiv). If you use any of the code here, please cite the above in your work.

## Reproducing paper results

1. Clone this repo to your local machine.
   
2. Ensure you have permission to access both [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/) and [eICU](https://www.physionet.org/content/eicu-crd/2.0/)  
(as well as [HiRID]() if you wish to run advanced experiments).
   
3. Download the [preprocessed datasets](https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/0) to the repo's `/data` subfolder:

    ```powershell
    wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/
    ```

4. To reproduce all experiments from the paper run:
   ```powershell
   python main.py
   ```

5. For real-time plotting of results via [tensorboard](https://www.tensorflow.org/tensorboard), run the following:
   ```powershell
   tensorboard --logdir=/results/tb_results/<tb_log_exp_name>
   ```

## Individual experiments

Individual experiments can be specified by the `--experiment` argument. Likewise, model architectures and continual learning strategies can be specified with the `--models` and `--strategies` args respectively e.g:

```powershell
python main.py --experiment region --models MLP CNN --strategies EWC Replay
```

For a list of permissable values, use the `--help` flag:

```powershell
python main.py --help
```

## Reproducibility

For ease of reproducibility, readability, and standardisation of results, we use the following tools in this project:

| Tool                        | Source               |
|-----------------------------|----------------------|
|ICU Data                     | [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/)<br> [eICU-CRD](https://www.physionet.org/content/eicu-crd/2.0/)<br> [HiRID](https://physionet.org/content/hirid/1.1.1/) |
| Data preprocessing / task definition | [FIDDLE](https://www.physionet.org/content/mimic-eicu-fiddle-feature/1.0.0/)<br> [HiRID-ICU-Benchmark](https://openreview.net/forum?id=SnC9rUeqiqd) |
|Continual Learning strategies| [Avalanche](https://avalanche.continualai.org/)

**Note:** Avalanche is  currently in its beta release. Development on this project was mostly completed using an earlier alpha build:
```
avalanche @ git+https://github.com/ContinualAI/avalanche.git@a2e2fb09f77eaecad8dbbe74b4b78ab737b7e464
```
If you encounter compatability issues related to Avalanche try using this specific version.