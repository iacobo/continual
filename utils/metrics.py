from typing import List, Union, Dict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task
from collections import defaultdict

# Custom binary prediction metrics using Avalanche
# For confusion_vector logic see:
# https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d

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

            confusion_vector = predicted_y / true_y

            true_positives = torch.sum(confusion_vector == 1).item()
            false_positives = torch.sum(confusion_vector == float('inf')).item()
            true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
            false_negatives = torch.sum(confusion_vector == 0).item()

            TPR = true_positives / (true_positives + false_negatives)
            TNR = true_negatives / (true_negatives + false_positives)
            self._mean_balancedaccuracy[task_labels].update(
                (TPR+TNR) / 2, len(predicted_y))
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
        return "Top1_RunningAcc_Epoch"



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