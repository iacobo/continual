[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

<!-- [![License](https://img.shields.io/github/license/iacobo/continual.svg)](https://opensource.org/licenses/MIT) -->


# Continual Learning of Longitudinal Health Records

Repo for reproducing the experiments in *Continual Learning of Longitudinal Health Records*.

## Setup

1. Clone this repo to your local machine.
2. Request access to the [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/) and [eICU-CRD](https://www.physionet.org/content/eicu-crd/2.0/) datasets.
3. Download the [preprocessed datasets](https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/0) to the `/<repo>/data` subfolder.
4. *(Recommended)* Create a new virtual environment:
   ```
   python -m pip install --user virtualenv
   python -m venv .venv
   ```
5. [Activate](https://docs.python.org/3/library/venv.html) virtual environment.
6. Upgrade the build tools and install dependencies:
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

Individual experiments can be specified with appropriate flags:

Flag           | Arg(s)      | Meaning
---------------|-------------|------------------------
`--experiment` | `region hospital age ethnicity` | Domain increment to use
`--outcome`    |`mortality_48h shock_4h shock_12h ARF_4h ARF_12h`       | Outcome to predict
`--models`     |`MLP CNN RNN LSTM Transformer`   | Model(s) to evaluate
`--strategies` |`Naive Cumulative EWC LwF SI GEM AGEM Replay GDumb` | Continual learning strategy(s) to evaluate
`--validate`   |             | Rerun hyper-parameter search
`--num_samples` |`<int>`         | Budget for hyper-parameter search

e.g.

```powershell
python main.py --experiment region --models CNN --strategies EWC Replay
```

Hyper-parameter search-space is specificed in `/config/config.py`. Pre-tuned settings are defined in `/config/best_config_<dataset>_<domain>_<outcome>.json`.

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
