![Python](https://badges.aleen42.com/src/python.svg) ![conda](https://img.shields.io/badge/%E2%80%8B-conda-%2344A833.svg?style=flat&logo=anaconda&logoColor=44A833) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![arXiv](https://img.shields.io/badge/arXiv-2112.11944-b31b1b.svg)](https://arxiv.org/abs/2112.11944)

# Continual Learning of Longitudinal Health Records

Repo for reproducing the experiments in [*Continual Learning of Longitudinal Health Records*](https://arxiv.org/abs/2112.11944) (2021). Release [v0.1](releases/v0.1) of the project corresponds to published results.

Experiments evaluate various continual learning strategies on standard ICU predictive tasks exhibiting covariate shift. Task outcomes are binary, and input data are multi-modal time-series from patient ICU admissions.

## Setup

1. Clone this repo to your local machine.
2. Request access to [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/) and [eICU-CRD](https://www.physionet.org/content/eicu-crd/2.0/).<sup>1</sup>
3. Download the [preprocessed datasets](https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/) to the `/data` subfolder.
4. Create and activate a virtual environment:

   ```posh
   conda env create -f environment.yml
   conda activate env-continual
   ```

## Results

To reproduce main results:

```posh
python3 main.py --train
```

Figures will be saved to `/results/figs`. Instructions to reproduce supplementary experiments can be found [here](/results/README.md). Bespoke experiments can be specified with appropriate flags e.g:

```posh
python3 main.py --domain_shift hospital --outcome mortality_48h --models CNN --strategies EWC Replay --validate --train
```

A complete list of available options can be found [here](/config/README.md) or with `python3 main.py --help`.

## Citation

If you use any of this code in your work, please reference us:

```latex
@misc{armstrong2021continual,
      title={Continual learning of longitudinal health records}, 
      author={J. Armstrong and D. Clifton},
      year={2021},
      eprint={2112.11944},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

---

### Notes

Note that Temporal Domain Incremental learning experiments require linkage with original MIMIC-III dataset. Requires downloading `ADMISSIONS.csv` from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) to the `/data/mimic3/` folder.

### Stack

For standardisation of ICU predictive task definitions, feature pre-processing, and Continual Learning method implementations, we use the following tools:

| Tool                        | Source               |
|-----------------------------|----------------------|
|ICU Data                     | [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/)<br> [eICU-CRD](https://www.physionet.org/content/eicu-crd/2.0/) |
| Data preprocessing / task definition | [FIDDLE](https://www.physionet.org/content/mimic-eicu-fiddle-feature/1.0.0/) |
|Continual Learning strategies| [Avalanche](https://avalanche.continualai.org/)
