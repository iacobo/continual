"""
Custom binary prediction metrics using Avalanche
https://github.com/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/05_evaluation.ipynb
"""

from typing import List, Union, Dict
from collections import defaultdict

import torch
from torch import Tensor, arange
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task

from sklearn.metrics import average_precision_score

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)

    Source: https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


# https://github.com/ContinualAI/avalanche/blob/master/avalanche/evaluation/metrics/mean_scores.py
# Use above for AUPRC etc templates.

class BalancedAccuracy(Metric[float]):
    """
    The BalancedAccuracy metric. This is a standalone metric.

    The metric keeps a dictionary of <task_label, balancedaccuracy value> pairs.
    and update the values through a running average over multiple
    <prediction, target> pairs of Tensors, provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average balancedaccuracy
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an balancedaccuracy value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the standalone BalancedAccuracy metric.

        By default this metric in its initial state will return an balancedaccuracy
        value of 0. The metric can be updated by using the `update` method
        while the running balancedaccuracy can be retrieved using the `result` method.
        """
        super().__init__()
        self._mean_balancedaccuracy = defaultdict(Mean)
        """
        The mean utility that will be used to store the running balancedaccuracy
        for each task label.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor,
               task_labels: Union[float, Tensor]) -> None:
        """
        Update the running balancedaccuracy given the true and predicted labels.
        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if Float, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.

        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError('Size mismatch for true_y and predicted_y tensors')

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError('Size mismatch for true_y and task_labels tensors')

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        if isinstance(task_labels, int):

            true_positives, false_positives, true_negatives, false_negatives = confusion(predicted_y, true_y)

            try:
                tpr = true_positives / (true_positives + false_negatives)
            except ZeroDivisionError:
                tpr = 0

            try:
                tnr = true_negatives / (true_negatives + false_positives)
            except ZeroDivisionError:
                tnr = 0
                
            self._mean_balancedaccuracy[task_labels].update(
                (tpr+tnr) / 2, len(predicted_y))
        elif isinstance(task_labels, Tensor):
            raise NotImplementedError
        else:
            raise ValueError(f"Task label type: {type(task_labels)}, "
                             f"expected int/float or Tensor")


    def result(self, task_label=None) -> Dict[int, float]:
        """
        Retrieves the running balancedaccuracy.

        Calling this method will not change the internal state of the metric.

        :param task_label: if None, return the entire dictionary of balanced accuracies
            for each task. Otherwise return the dictionary
            `{task_label: balancedaccuracy}`.
        :return: A dict of running balanced accuracies for each task label,
            where each value is a float value between 0 and 1.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            return {k: v.result() for k, v in self._mean_balancedaccuracy.items()}
        else:
            return {task_label: self._mean_balancedaccuracy[task_label].result()}


    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            self._mean_balancedaccuracy = defaultdict(Mean)
        else:
            self._mean_balancedaccuracy[task_label].reset()

class BalancedAccuracyPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all balanced accuracies plugin metrics
    """
    def __init__(self, reset_at, emit_at, mode):
        self._balancedaccuracy = BalancedAccuracy()
        super(BalancedAccuracyPluginMetric, self).__init__(
            self._balancedaccuracy, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def reset(self, strategy=None) -> None:
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._balancedaccuracy.update(strategy.mb_output, strategy.mb_y, task_labels)


class MinibatchBalancedAccuracy(BalancedAccuracyPluginMetric):
    """
    The minibatch plugin balancedaccuracy metric.
    This metric only works at training time.

    This metric computes the average balancedaccuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochBalancedAccuracy` instead.
    """
    def __init__(self):
        """
        Creates an instance of the MinibatchBalancedAccuracy metric.
        """
        super(MinibatchBalancedAccuracy, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "BalAcc_MB"



class EpochBalancedAccuracy(BalancedAccuracyPluginMetric):
    """
    The average balancedaccuracy over a single training epoch.
    This plugin metric only works at training time.

    The balancedaccuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochBalancedAccuracy metric.
        """

        super(EpochBalancedAccuracy, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "BalAcc_Epoch"



class RunningEpochBalancedAccuracy(BalancedAccuracyPluginMetric):
    """
    The average balancedaccuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the balancedaccuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochBalancedAccuracy metric.
        """

        super(RunningEpochBalancedAccuracy, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train')

    def __str__(self):
        return "RunningBalAcc_Epoch"



class ExperienceBalancedAccuracy(BalancedAccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average balancedaccuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceBalancedAccuracy metric
        """
        super(ExperienceBalancedAccuracy, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return "BalAcc_Exp"



class StreamBalancedAccuracy(BalancedAccuracyPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average balancedaccuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamBalancedAccuracy metric
        """
        super(StreamBalancedAccuracy, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return "BalAcc_Stream"



class TrainedExperienceBalancedAccuracy(BalancedAccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    balancedaccuracy for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceBalancedAccuracy metric by first 
        constructing BalancedAccuracyPluginMetric
        """
        super(TrainedExperienceBalancedAccuracy, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience 
        BalancedAccuracyPluginMetric.reset(self, strategy)
        return BalancedAccuracyPluginMetric.after_training_exp(self, strategy)

        
    def update(self, strategy):
        """
        Only update the balancedaccuracy with results from experiences that have been 
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            BalancedAccuracyPluginMetric.update(self, strategy)


    def __str__(self):
        return "BalancedAccuracy_On_Trained_Experiences"



def balancedaccuracy_metrics(*, 
                     minibatch=False,
                     epoch=False,
                     epoch_running=False,
                     experience=False,
                     stream=False,
                     trained_experience=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch balancedaccuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch balancedaccuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch balancedaccuracy at training time.
    :param experience: If True, will return a metric able to log
        the balancedaccuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the balancedaccuracy averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation balancedaccuracy only for experiences that the
        model has been trained on         

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchBalancedAccuracy())

    if epoch:
        metrics.append(EpochBalancedAccuracy())

    if epoch_running:
        metrics.append(RunningEpochBalancedAccuracy())

    if experience:
        metrics.append(ExperienceBalancedAccuracy())

    if stream:
        metrics.append(StreamBalancedAccuracy())

    if trained_experience:
        metrics.append(TrainedExperienceBalancedAccuracy())

    return metrics


class Sensitivity(Metric[float]):
    """
    The Sensitivity metric. This is a standalone metric.

    The metric keeps a dictionary of <task_label, Sensitivity value> pairs.
    and update the values through a running average over multiple
    <prediction, target> pairs of Tensors, provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average Sensitivity
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an Sensitivity value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the standalone Sensitivity metric.

        By default this metric in its initial state will return an Sensitivity
        value of 0. The metric can be updated by using the `update` method
        while the running Sensitivity can be retrieved using the `result` method.
        """
        super().__init__()
        self._mean_Sensitivity = defaultdict(Mean)
        """
        The mean utility that will be used to store the running Sensitivity
        for each task label.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor,
               task_labels: Union[float, Tensor]) -> None:
        """
        Update the running Sensitivity given the true and predicted labels.
        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if Float, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.

        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError('Size mismatch for true_y and predicted_y tensors')

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError('Size mismatch for true_y and task_labels tensors')

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        if isinstance(task_labels, int):

            true_positives, false_positives, true_negatives, false_negatives = confusion(predicted_y, true_y)

            try:
                tpr = true_positives / (true_positives + false_negatives)
            except ZeroDivisionError:
                tpr = 0
                
            self._mean_Sensitivity[task_labels].update(
                tpr, len(predicted_y))
        elif isinstance(task_labels, Tensor):
            raise NotImplementedError
        else:
            raise ValueError(f"Task label type: {type(task_labels)}, "
                             f"expected int/float or Tensor")


    def result(self, task_label=None) -> Dict[int, float]:
        """
        Retrieves the running Sensitivity.

        Calling this method will not change the internal state of the metric.

        :param task_label: if None, return the entire dictionary of sensitivities
            for each task. Otherwise return the dictionary
            `{task_label: Sensitivity}`.
        :return: A dict of running sensitivities for each task label,
            where each value is a float value between 0 and 1.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            return {k: v.result() for k, v in self._mean_Sensitivity.items()}
        else:
            return {task_label: self._mean_Sensitivity[task_label].result()}


    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            self._mean_Sensitivity = defaultdict(Mean)
        else:
            self._mean_Sensitivity[task_label].reset()

class SensitivityPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all sensitivities plugin metrics
    """
    def __init__(self, reset_at, emit_at, mode):
        self._Sensitivity = Sensitivity()
        super(SensitivityPluginMetric, self).__init__(
            self._Sensitivity, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def reset(self, strategy=None) -> None:
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._Sensitivity.update(strategy.mb_output, strategy.mb_y, task_labels)


class MinibatchSensitivity(SensitivityPluginMetric):
    """
    The minibatch plugin Sensitivity metric.
    This metric only works at training time.

    This metric computes the average Sensitivity over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochSensitivity` instead.
    """
    def __init__(self):
        """
        Creates an instance of the MinibatchSensitivity metric.
        """
        super(MinibatchSensitivity, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "Sens_MB"



class EpochSensitivity(SensitivityPluginMetric):
    """
    The average Sensitivity over a single training epoch.
    This plugin metric only works at training time.

    The Sensitivity will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochSensitivity metric.
        """

        super(EpochSensitivity, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "Sens_Epoch"



class RunningEpochSensitivity(SensitivityPluginMetric):
    """
    The average Sensitivity across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the Sensitivity averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochSensitivity metric.
        """

        super(RunningEpochSensitivity, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train')

    def __str__(self):
        return "RunningSens_Epoch"



class ExperienceSensitivity(SensitivityPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average Sensitivity over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceSensitivity metric
        """
        super(ExperienceSensitivity, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return "Sens_Exp"



class StreamSensitivity(SensitivityPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average Sensitivity over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamSensitivity metric
        """
        super(StreamSensitivity, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return "Sens_Stream"



class TrainedExperienceSensitivity(SensitivityPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    Sensitivity for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceSensitivity metric by first 
        constructing SensitivityPluginMetric
        """
        super(TrainedExperienceSensitivity, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience 
        SensitivityPluginMetric.reset(self, strategy)
        return SensitivityPluginMetric.after_training_exp(self, strategy)

        
    def update(self, strategy):
        """
        Only update the Sensitivity with results from experiences that have been 
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            SensitivityPluginMetric.update(self, strategy)


    def __str__(self):
        return "Sensitivity_On_Trained_Experiences"



def Sensitivity_metrics(*, 
                     minibatch=False,
                     epoch=False,
                     epoch_running=False,
                     experience=False,
                     stream=False,
                     trained_experience=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch Sensitivity at training time.
    :param epoch: If True, will return a metric able to log
        the epoch Sensitivity at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch Sensitivity at training time.
    :param experience: If True, will return a metric able to log
        the Sensitivity on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the Sensitivity averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation Sensitivity only for experiences that the
        model has been trained on         

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchSensitivity())

    if epoch:
        metrics.append(EpochSensitivity())

    if epoch_running:
        metrics.append(RunningEpochSensitivity())

    if experience:
        metrics.append(ExperienceSensitivity())

    if stream:
        metrics.append(StreamSensitivity())

    if trained_experience:
        metrics.append(TrainedExperienceSensitivity())

    return metrics


class Specificity(Metric[float]):
    """
    The Specificity metric. This is a standalone metric.

    The metric keeps a dictionary of <task_label, Specificity value> pairs.
    and update the values through a running average over multiple
    <prediction, target> pairs of Tensors, provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average Specificity
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an Specificity value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the standalone Specificity metric.

        By default this metric in its initial state will return an Specificity
        value of 0. The metric can be updated by using the `update` method
        while the running Specificity can be retrieved using the `result` method.
        """
        super().__init__()
        self._mean_Specificity = defaultdict(Mean)
        """
        The mean utility that will be used to store the running Specificity
        for each task label.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor,
               task_labels: Union[float, Tensor]) -> None:
        """
        Update the running Specificity given the true and predicted labels.
        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if Float, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.

        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError('Size mismatch for true_y and predicted_y tensors')

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError('Size mismatch for true_y and task_labels tensors')

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        if isinstance(task_labels, int):

            true_positives, false_positives, true_negatives, false_negatives = confusion(predicted_y, true_y)

            try:
                tnr = true_negatives / (true_negatives + false_positives)
            except ZeroDivisionError:
                tnr = 0
                
            self._mean_Specificity[task_labels].update(
                tnr, len(predicted_y))
        elif isinstance(task_labels, Tensor):
            raise NotImplementedError
        else:
            raise ValueError(f"Task label type: {type(task_labels)}, "
                             f"expected int/float or Tensor")


    def result(self, task_label=None) -> Dict[int, float]:
        """
        Retrieves the running Specificity.

        Calling this method will not change the internal state of the metric.

        :param task_label: if None, return the entire dictionary of specificities
            for each task. Otherwise return the dictionary
            `{task_label: Specificity}`.
        :return: A dict of running specificities for each task label,
            where each value is a float value between 0 and 1.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            return {k: v.result() for k, v in self._mean_Specificity.items()}
        else:
            return {task_label: self._mean_Specificity[task_label].result()}


    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            self._mean_Specificity = defaultdict(Mean)
        else:
            self._mean_Specificity[task_label].reset()

class SpecificityPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all specificities plugin metrics
    """
    def __init__(self, reset_at, emit_at, mode):
        self._Specificity = Specificity()
        super(SpecificityPluginMetric, self).__init__(
            self._Specificity, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def reset(self, strategy=None) -> None:
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._Specificity.update(strategy.mb_output, strategy.mb_y, task_labels)


class MinibatchSpecificity(SpecificityPluginMetric):
    """
    The minibatch plugin Specificity metric.
    This metric only works at training time.

    This metric computes the average Specificity over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochSpecificity` instead.
    """
    def __init__(self):
        """
        Creates an instance of the MinibatchSpecificity metric.
        """
        super(MinibatchSpecificity, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "Spec_MB"



class EpochSpecificity(SpecificityPluginMetric):
    """
    The average Specificity over a single training epoch.
    This plugin metric only works at training time.

    The Specificity will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochSpecificity metric.
        """

        super(EpochSpecificity, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "Spec_Epoch"



class RunningEpochSpecificity(SpecificityPluginMetric):
    """
    The average Specificity across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the Specificity averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochSpecificity metric.
        """

        super(RunningEpochSpecificity, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train')

    def __str__(self):
        return "RunningSpec_Epoch"



class ExperienceSpecificity(SpecificityPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average Specificity over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceSpecificity metric
        """
        super(ExperienceSpecificity, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return "Spec_Exp"



class StreamSpecificity(SpecificityPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average Specificity over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamSpecificity metric
        """
        super(StreamSpecificity, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return "Spec_Stream"



class TrainedExperienceSpecificity(SpecificityPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    Specificity for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceSpecificity metric by first 
        constructing SpecificityPluginMetric
        """
        super(TrainedExperienceSpecificity, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience 
        SpecificityPluginMetric.reset(self, strategy)
        return SpecificityPluginMetric.after_training_exp(self, strategy)

        
    def update(self, strategy):
        """
        Only update the Specificity with results from experiences that have been 
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            SpecificityPluginMetric.update(self, strategy)


    def __str__(self):
        return "Specificity_On_Trained_Experiences"



def Specificity_metrics(*, 
                     minibatch=False,
                     epoch=False,
                     epoch_running=False,
                     experience=False,
                     stream=False,
                     trained_experience=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch Specificity at training time.
    :param epoch: If True, will return a metric able to log
        the epoch Specificity at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch Specificity at training time.
    :param experience: If True, will return a metric able to log
        the Specificity on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the Specificity averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation Specificity only for experiences that the
        model has been trained on         

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchSpecificity())

    if epoch:
        metrics.append(EpochSpecificity())

    if epoch_running:
        metrics.append(RunningEpochSpecificity())

    if experience:
        metrics.append(ExperienceSpecificity())

    if stream:
        metrics.append(StreamSpecificity())

    if trained_experience:
        metrics.append(TrainedExperienceSpecificity())

    return metrics


class Precision(Metric[float]):
    """
    The Precision metric. This is a standalone metric.

    The metric keeps a dictionary of <task_label, Precision value> pairs.
    and update the values through a running average over multiple
    <prediction, target> pairs of Tensors, provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average Precision
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an Precision value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the standalone Precision metric.

        By default this metric in its initial state will return a Precision
        value of 0. The metric can be updated by using the `update` method
        while the running Precision can be retrieved using the `result` method.
        """
        super().__init__()
        self._mean_Precision = defaultdict(Mean)
        """
        The mean utility that will be used to store the running Precision
        for each task label.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor,
               task_labels: Union[float, Tensor]) -> None:
        """
        Update the running Precision given the true and predicted labels.
        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if Float, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.

        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError('Size mismatch for true_y and predicted_y tensors')

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError('Size mismatch for true_y and task_labels tensors')

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        if isinstance(task_labels, int):

            true_positives, false_positives, true_negatives, false_negatives = confusion(predicted_y, true_y)

            try:
                ppv = true_positives / (true_positives + false_positives)
            except ZeroDivisionError:
                ppv = 0
                
            self._mean_Precision[task_labels].update(
                ppv, len(predicted_y))
        elif isinstance(task_labels, Tensor):
            raise NotImplementedError
        else:
            raise ValueError(f"Task label type: {type(task_labels)}, "
                             f"expected int/float or Tensor")


    def result(self, task_label=None) -> Dict[int, float]:
        """
        Retrieves the running Precision.

        Calling this method will not change the internal state of the metric.

        :param task_label: if None, return the entire dictionary of precisions
            for each task. Otherwise return the dictionary
            `{task_label: Precision}`.
        :return: A dict of running precisions for each task label,
            where each value is a float value between 0 and 1.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            return {k: v.result() for k, v in self._mean_Precision.items()}
        else:
            return {task_label: self._mean_Precision[task_label].result()}


    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            self._mean_Precision = defaultdict(Mean)
        else:
            self._mean_Precision[task_label].reset()

class PrecisionPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all precisions plugin metrics
    """
    def __init__(self, reset_at, emit_at, mode):
        self._Precision = Precision()
        super(PrecisionPluginMetric, self).__init__(
            self._Precision, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def reset(self, strategy=None) -> None:
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._Precision.update(strategy.mb_output, strategy.mb_y, task_labels)


class MinibatchPrecision(PrecisionPluginMetric):
    """
    The minibatch plugin Precision metric.
    This metric only works at training time.

    This metric computes the average Precision over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochPrecision` instead.
    """
    def __init__(self):
        """
        Creates an instance of the MinibatchPrecision metric.
        """
        super(MinibatchPrecision, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "Prec_MB"



class EpochPrecision(PrecisionPluginMetric):
    """
    The average Precision over a single training epoch.
    This plugin metric only works at training time.

    The Precision will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochPrecision metric.
        """

        super(EpochPrecision, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "Prec_Epoch"



class RunningEpochPrecision(PrecisionPluginMetric):
    """
    The average Precision across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the Precision averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochPrecision metric.
        """

        super(RunningEpochPrecision, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train')

    def __str__(self):
        return "RunningPrec_Epoch"



class ExperiencePrecision(PrecisionPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average Precision over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperiencePrecision metric
        """
        super(ExperiencePrecision, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return "Prec_Exp"



class StreamPrecision(PrecisionPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average Precision over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamPrecision metric
        """
        super(StreamPrecision, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return "Prec_Stream"



class TrainedExperiencePrecision(PrecisionPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    Precision for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperiencePrecision metric by first 
        constructing PrecisionPluginMetric
        """
        super(TrainedExperiencePrecision, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience 
        PrecisionPluginMetric.reset(self, strategy)
        return PrecisionPluginMetric.after_training_exp(self, strategy)

        
    def update(self, strategy):
        """
        Only update the Precision with results from experiences that have been 
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            PrecisionPluginMetric.update(self, strategy)


    def __str__(self):
        return "Precision_On_Trained_Experiences"



def Precision_metrics(*, 
                     minibatch=False,
                     epoch=False,
                     epoch_running=False,
                     experience=False,
                     stream=False,
                     trained_experience=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch Precision at training time.
    :param epoch: If True, will return a metric able to log
        the epoch Precision at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch Precision at training time.
    :param experience: If True, will return a metric able to log
        the Precision on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the Precision averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation Precision only for experiences that the
        model has been trained on         

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchPrecision())

    if epoch:
        metrics.append(EpochPrecision())

    if epoch_running:
        metrics.append(RunningEpochPrecision())

    if experience:
        metrics.append(ExperiencePrecision())

    if stream:
        metrics.append(StreamPrecision())

    if trained_experience:
        metrics.append(TrainedExperiencePrecision())

    return metrics


class AUPRC(Metric[float]):
    """
    The AUPRC metric. This is a standalone metric.

    The metric keeps a dictionary of <task_label, AUPRC value> pairs.
    and update the values through a running average over multiple
    <prediction, target> pairs of Tensors, provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average AUPRC
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an AUPRC value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the standalone AUPRC metric.

        By default this metric in its initial state will return a AUPRC
        value of 0. The metric can be updated by using the `update` method
        while the running AUPRC can be retrieved using the `result` method.
        """
        super().__init__()
        self._mean_AUPRC = defaultdict(Mean)
        """
        The mean utility that will be used to store the running AUPRC
        for each task label.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor,
               task_labels: Union[float, Tensor]) -> None:
        """
        Update the running AUPRC given the true and predicted labels.
        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if Float, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.

        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError('Size mismatch for true_y and predicted_y tensors')

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError('Size mismatch for true_y and task_labels tensors')

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        assert (
            len(predicted_y.size()) == 2
        ), "Predictions need to be logits or scores, not labels"

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        scores = predicted_y[arange(len(true_y)), true_y]

        if isinstance(task_labels, int):
            self._mean_AUPRC[task_labels].update(
                average_precision_score(true_y, scores), len(predicted_y))
        elif isinstance(task_labels, Tensor):
            raise NotImplementedError
        else:
            raise ValueError(f"Task label type: {type(task_labels)}, "
                             f"expected int/float or Tensor")


    def result(self, task_label=None) -> Dict[int, float]:
        """
        Retrieves the running AUPRC.

        Calling this method will not change the internal state of the metric.

        :param task_label: if None, return the entire dictionary of AUPRCs
            for each task. Otherwise return the dictionary
            `{task_label: AUPRC}`.
        :return: A dict of running AUPRCs for each task label,
            where each value is a float value between 0 and 1.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            return {k: v.result() for k, v in self._mean_AUPRC.items()}
        else:
            return {task_label: self._mean_AUPRC[task_label].result()}


    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            self._mean_AUPRC = defaultdict(Mean)
        else:
            self._mean_AUPRC[task_label].reset()

class AUPRCPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all AUPRCs plugin metrics
    """
    def __init__(self, reset_at, emit_at, mode):
        self._AUPRC = AUPRC()
        super(AUPRCPluginMetric, self).__init__(
            self._AUPRC, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def reset(self, strategy=None) -> None:
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._AUPRC.update(strategy.mb_output, strategy.mb_y, task_labels)


class MinibatchAUPRC(AUPRCPluginMetric):
    """
    The minibatch plugin AUPRC metric.
    This metric only works at training time.

    This metric computes the average AUPRC over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAUPRC` instead.
    """
    def __init__(self):
        """
        Creates an instance of the MinibatchAUPRC metric.
        """
        super(MinibatchAUPRC, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "AUPRC_MB"



class EpochAUPRC(AUPRCPluginMetric):
    """
    The average AUPRC over a single training epoch.
    This plugin metric only works at training time.

    The AUPRC will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochAUPRC metric.
        """

        super(EpochAUPRC, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "AUPRC_Epoch"



class RunningEpochAUPRC(AUPRCPluginMetric):
    """
    The average AUPRC across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the AUPRC averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochAUPRC metric.
        """

        super(RunningEpochAUPRC, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train')

    def __str__(self):
        return "RunningAUPRC_Epoch"



class ExperienceAUPRC(AUPRCPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average AUPRC over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAUPRC metric
        """
        super(ExperienceAUPRC, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return "AUPRC_Exp"



class StreamAUPRC(AUPRCPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average AUPRC over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamAUPRC metric
        """
        super(StreamAUPRC, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return "AUPRC_Stream"



class TrainedExperienceAUPRC(AUPRCPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    AUPRC for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceAUPRC metric by first 
        constructing AUPRCPluginMetric
        """
        super(TrainedExperienceAUPRC, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience 
        AUPRCPluginMetric.reset(self, strategy)
        return AUPRCPluginMetric.after_training_exp(self, strategy)

        
    def update(self, strategy):
        """
        Only update the AUPRC with results from experiences that have been 
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            AUPRCPluginMetric.update(self, strategy)


    def __str__(self):
        return "AUPRC_On_Trained_Experiences"



def AUPRC_metrics(*, 
                     minibatch=False,
                     epoch=False,
                     epoch_running=False,
                     experience=False,
                     stream=False,
                     trained_experience=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch AUPRC at training time.
    :param epoch: If True, will return a metric able to log
        the epoch AUPRC at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch AUPRC at training time.
    :param experience: If True, will return a metric able to log
        the AUPRC on each evaluation experience.
    :param stream: If True, will return a metric able to logAUPRCperiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation AUPRC only for experiences that the
        model has been trained on         

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchAUPRC())

    if epoch:
        metrics.append(EpochAUPRC())

    if epoch_running:
        metrics.append(RunningEpochAUPRC())

    if experience:
        metrics.append(ExperienceAUPRC())

    if stream:
        metrics.append(StreamAUPRC())

    if trained_experience:
        metrics.append(TrainedExperienceAUPRC())

    return metrics



__all__ = [
    'BalancedAccuracy',
    'MinibatchBalancedAccuracy',
    'EpochBalancedAccuracy',
    'RunningEpochBalancedAccuracy',
    'ExperienceBalancedAccuracy',
    'StreamBalancedAccuracy',
    'TrainedExperienceBalancedAccuracy',
    'balancedaccuracy_metrics'
]
