# Continual Learning of Longitudinal Health Records / ICU EHR

Repo for reproducing the experiments in *Continual Learning of Longitudinal Health Records* (arxiv). If you use any of the code here, please cite the above in your work.

### Instructions

1. Clone this repo to your local machine.
   
2. Ensure you have permission to access both the [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/) and [eICU](https://www.physionet.org/content/eicu-crd/2.0/) datasets (as well as [HiRID]() if you wish to run advanced experiments).
   
3. Download the [preprocessed datasets](https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/0) to the repo's `/data` subfolder:

    ```
    wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/
    ```

4. To run all experiments with the hyperparameters quoted in the paper:
   ```
   python main.py
   ```
5. To run all experiments with hyperparameter tuning (note this requires `ray tune`):
   ```
   python main.py --validate
   ```
6. Individual experiments can be specified by the `--experiment` argument. Likewise, a model architectures and continual learning strategies can be specified with the `--models` and `--strategies` args respectively e.g:
   ```
   python main.py --experiment region --models MLP CNN --strategies EWC Replay
   ```

   For a list of permissable values, run the `--help` flag:

   ```
   >>> python main.py -h

   usage: main.py [-h] [--validate] [--experiment {time_month,time_season,time_year,region,hospital,age,sex,ethnicity}] [--strategies {Naive,Cumulative,EWC,SI,LwF,Replay,GEM}] [--models {MLP,CNN,RNN,LSTM}]
   
   optional arguments:
      -h, --help            show this help message and exit
      --validate            Run hyperparameter optimisation routine during training.
      --experiment {time_month,time_season,time_year,region,hospital,age,sex,ethnicity}
                            Name of experiment to run.
      --strategies {Naive,Cumulative,EWC,SI,LwF,Replay,GEM}
                            Continual learning strategy(s) to evaluate.
      --models {MLP,CNN,RNN,LSTM}
                            Model(s) to evaluate.
   ```

## Reproducibility

For ease of reproducibility, code readability, and standardisation of results, we use the following tools in this project:

| Tool                        | Source               |
|-----------------------------|----------------------|
|ICU Data                     | [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/)<br> [eICU-CRD](https://www.physionet.org/content/eicu-crd/2.0/)<br> [HiRID](https://physionet.org/content/hirid/1.1.1/) |
| Data preprocessing / task definition | [FIDDLE](https://www.physionet.org/content/mimic-eicu-fiddle-feature/1.0.0/)<br> [HiRID-ICU-Benchmark](https://openreview.net/forum?id=SnC9rUeqiqd) |
|Continual Learning strategies| [Avalanche](https://avalanche.continualai.org/)