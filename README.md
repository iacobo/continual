# Continual Learning of Longitudinal Health Records / ICU EHR

Repo for reproducing the experiments in *Continual Learning of Longitudinal Health Records* (arxiv). If you use any of the code here, please cite the above in your work:

```latex
@article{armstrong2021continual,
  title={Continual Learning of Longitudinal Health Records},
  author={Armstrong, Jacob},
  year={2021}
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
   Figures will be saved to `/results/figs`. For real-time plotting of results via [tensorboard](https://www.tensorflow.org/tensorboard), run:
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

## Hyperparameters

Experiments use the hyperparameter settings found in `/config/best_config_<dataset>_<domain>_<outcome>.json`   

If `--validate`, experiments will run a hyperparameter sweep over the search-space specificed in `/config/config.py` instead.

## Reproducibility

For standardisation of task definitions, feature pre-processing, and model implementations, we use the following tools:

| Tool                        | Source               |
|-----------------------------|----------------------|
|ICU Data                     | [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/)<br> [eICU-CRD](https://www.physionet.org/content/eicu-crd/2.0/)<br> [HiRID](https://physionet.org/content/hirid/1.1.1/) |
| Data preprocessing / task definition | [FIDDLE](https://www.physionet.org/content/mimic-eicu-fiddle-feature/1.0.0/)<br> [HiRID-ICU-Benchmark](https://openreview.net/forum?id=SnC9rUeqiqd) |
|Continual Learning strategies| [Avalanche](https://avalanche.continualai.org/)\*\*


## Project structure

- `main.py` (Main training program)
- `test.py` (Test suite)
- config
  - `config.py` (Hyperparameter search space configuration)
- data
  - FIDDLE_eicu (Pre-processed eICU-CRD dataset)
  - FIDDLE_mimic3 (Pre-processed MIMIC-III dataset)
- results
  - figs (Plotted results)
  - log
  - metrics (Results of experiments)
- utils
  - `data_processing.py` (Code to load and pre-process datasets, split datasets along task boundaries)
  - `models.py` (Model definitions, continual learning strategies)
  - `training.py` (Functions for performing hyper-parameter optimisation, training, and evaluation of models)
  - `plotting.py` (Functions to plot results)


---

<sup>

### Note

\* As well as [HiRID](https://physionet.org/content/hirid/1.1.1/) if you wish to run advanced experiments.

\*\* Avalanche is currently in its beta release. Development on this project was mostly completed using an earlier alpha build (0.3.0). Some method implementations and results may be inconsistent across versions.

\* Note that Temporal Domain Incremental learning experiments require linkage with original MIMIC-III and eICU-CRD datasets. Scripts to post-process can be found in `.../.../....py`

</sup>