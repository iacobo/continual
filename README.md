[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

<!-- [![License](https://img.shields.io/github/license/iacobo/continual.svg)](https://opensource.org/licenses/MIT) -->


# Continual Learning of Longitudinal Health Records

Repo for reproducing the experiments in *Continual Learning of Longitudinal Health Records*.

## Setup

1. Clone this repo to your local machine.
2. Request access to the [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/) and [eICU-CRD](https://www.physionet.org/content/eicu-crd/2.0/) datasets.
3. Download the [preprocessed datasets](https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/0) to the `/<repo>/data` subfolder.
4. Upgrade the build tools and install dependencies:
   ```
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

## Results

To rerun all experiments from the paper:
```powershell
python main.py
```
Figures will be saved to `/results/figs`.

## Individual experiments

Individual experiments can be specified by the `--experiment` (domain increment to use) and `--outcome` (outcome to predict) arguments. Likewise, model architectures and continual learning strategies can be specified with the `--models` and `--strategies` args respectively e.g:

```powershell
python main.py --experiment region --models CNN --strategies EWC Replay
```

For a list of permissable values, use the `--help` flag:

```powershell
python main.py --help
```

If `--validate`, experiments will run a hyperparameter sweep over the search-space specificed in `/config/config.py` using `--num_samples` samples. Otherwise, experiments use the hyperparameter settings discovered in the original paper (located in`/config/best_config_<dataset>_<domain>_<outcome>.json`). 

For real-time plotting of results via [tensorboard](https://www.tensorflow.org/tensorboard), run:
```powershell
tensorboard --logdir=/results/log/tb_results/<tb_log_exp_name>
```


## Citation

If you use any of this code in your work, please reference us:

```latex
@article{armstrong2021continual,
  title={Continual Learning of Longitudinal Health Records},
  author={Armstrong, Jacob},
  year={2021}
}
```

## Stack

For standardisation of ICU predictive task definitions, feature pre-processing, and Continual Learning method implementations, we use the following tools:

| Tool                        | Source               |
|-----------------------------|----------------------|
|ICU Data                     | [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/)<br> [eICU-CRD](https://www.physionet.org/content/eicu-crd/2.0/)<br> [HiRID](https://physionet.org/content/hirid/1.1.1/) |
| Data preprocessing / task definition | [FIDDLE](https://www.physionet.org/content/mimic-eicu-fiddle-feature/1.0.0/)<br> [HiRID-ICU-Benchmark](https://openreview.net/forum?id=SnC9rUeqiqd) |
|Continual Learning strategies| [Avalanche](https://avalanche.continualai.org/)

---

![Python versions](https://img.shields.io/badge/python-3.7+-1177AA.svg?logo=python) [![Tests](https://github.com/iacobo/continual/workflows/Tests/badge.svg)](https://github.com/iacobo/continual/actions)

\* As well as [HiRID](https://physionet.org/content/hirid/1.1.1/) if you wish to run advanced experiments.  
\* Note that Temporal Domain Incremental learning experiments require linkage with original MIMIC-III and eICU-CRD datasets. Scripts to post-process can be found in `.../.../....py`
