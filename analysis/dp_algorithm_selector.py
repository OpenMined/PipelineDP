import pipeline_dp
from dataclasses import dataclass
from typing import List

@dataclass
class DPStrategy:
    noise_kind: pipeline_dp.NoiseKind
    partition_selection_strategy: pipeline_dp.PartitionSelectionStrategy
    return_thresholding_values: bool  # todo better name


class DPAlgorithmSelector:

    def __init__(self, metric: pipeline_dp.Metric, is_public_partitions:bool):
        self._metric = metric
        self._is_public_partitions = is_public_partitions

    def get_dp_strategy(self, l0: int, linf: float) -> DPStrategy:
        pass

    def does_support_return_thresholding_values(self):
        return True

    def supported_noise_kind(self)->List[pipeline_dp.NoiseKind]:
        return [pipeline_dp.NoiseKind.LAPLACE, pipeline_dp.NoiseKind.GAUSSIAN]

    # def supported_