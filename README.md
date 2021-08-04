# Continual Learning of Longitudinal Health Records / ICU EHR

Repo for reproducing the experiments in *Continual Learning of Longitudinal Health Records* (arxiv). If you use any of the code here, please cite the above in your work.

### Instructions

1. Clone this repo to your local machine.
2. Ensure you have permission to access both the [MIMIC-III](https://www.physionet.org/content/mimiciii/1.4/) and [eICU](https://www.physionet.org/content/eicu-crd/2.0/) datasets.
3. Download the [preprocessed datasets](https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/0) to the repo's `/data` subfolder:

    ```
    wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/
    ```

4. To run all experiments with the hyperparameters quoted in the paper:
   ```
   python main.py
   ```
5. To run all experiments with hyperparameter tuning:
   ```
   python main.py validate=True
   ```
6. Individual experiments can be specified by the `experiment` argument. Likewise, a subset of model architectures and continual learning mechanisms can be specified with the `models` and `strategies` args respectively e.g:
   ```
   python main.py experiment=region models=['MLP','CNN'] strategies=['EWC','Replay']
   ```
   INCLUDE HELP FOR FUNCTION SHOWING POSSIBLE ARGUMENTS