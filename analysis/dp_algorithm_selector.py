import pipeline_dp
from dataclasses import dataclass


@dataclass
class DPStrategy:
    noise_kind: pipeline_dp.NoiseKind
    partition_selection_strategy: pipeline_dp.PartitionSelectionStrategy
    return_thresholding_values: bool  # todo better name


class DPAlgorithmSelector:

    def __init__(self, metric: pipeline_dp.Metric):
        self._metric = metric

    def get_dp_strategy(self, l0: int, linf: float) -> DPStrategy:
        pass
