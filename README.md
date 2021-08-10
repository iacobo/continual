# Continual Learning of Longitudinal Health Records / ICU EHR

Repo for reproducing the experiments in *Continual Learning of Longitudinal Health Records* (arxiv). If you use any of the code here, please cite the above in your work:

```latex
@article{armstrong2021continual,
  title={Continual Learning of Longitudinal Health Records},
  author={Armstrong, Jacob}
}
```

## Reproducing paper results

1. Clone this repo to your local machine.
   
2. Ensure you have permission to access both [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/) and [eICU](https://www.physionet.org/content/eicu-crd/2.0/).\*
   
3. Download the [preprocessed FIDDLE datasets](https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/0) to the repo's `/data` subfolder.

4. To rerun all experiments from the paper:
   ```powershell
   python main.py
   ```
   Figures will be saved to the `/results/figs`. For real-time plotting of results via [tensorboard](https://www.tensorflow.org/tensorboard), run:
   ```powershell
   tensorboard --logdir=/results/tb_results/<tb_log_exp_name>
   ```

## Individual experiments

Individual experiments can be specified by the `--experiment` argument. Likewise, model architectures and continual learning strategies can be specified with the `--models` and `--strategies` args respectively e.g:

```powershell
python main.py --experiment region --models CNN --strategies EWC Replay
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
|Continual Learning strategies| [Avalanche](https://avalanche.continualai.org/)\*\*

---

<sup>

### Note

\* As well as [HiRID](https://physionet.org/content/hirid/1.1.1/) if you wish to run advanced experiments.

\*\* Avalanche is currently in its beta release. Development on this project was mostly completed using an earlier alpha build:
```
avalanche @ git+https://github.com/ContinualAI/avalanche.git@a2e2fb09f77eaecad8dbbe74b4b78ab737b7e464
```
Since `pip` versioning may have been irregular during this period, if you encounter compatability issues related to Avalanche try installing this specific version.

</sup>