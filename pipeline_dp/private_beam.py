from apache_beam.transforms import ptransform
from abc import abstractmethod
from typing import Callable, Optional
from apache_beam import pvalue

import pipeline_dp
from pipeline_dp import aggregate_params, budget_accounting


class PrivateTransform(ptransform.PTransform):
    """Abstract class for PrivateTransforms."""

    def __init__(self, return_private: bool, label: Optional[str] = None):
        super().__init__(label)
        self._return_private = return_private
        self._budget_accountant = None
        self._privacy_id_extractor = None

    def set_additional_parameters(
            self, budget_accountant: budget_accounting.BudgetAccountant,
            privacy_id_extractor: Callable):
        """Set the additional parameters needed for the private transform."""
        self._budget_accountant = budget_accountant
        self._privacy_id_extractor = privacy_id_extractor

    @abstractmethod
    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        pass


class PrivateCollection:
    """Private counterpart for PCollection."""

    def __init__(self, pcol: pvalue.PCollection,
                 budget_accountant: budget_accounting.BudgetAccountant,
                 privacy_id_extractor: Callable):
        self._pcol = pcol
        self._budget_accountant = budget_accountant
        self._privacy_id_extractor = privacy_id_extractor

    def __or__(self, private_transform: PrivateTransform):
        if not isinstance(private_transform, PrivateTransform):
            raise TypeError(
                "private_transform should be of type PrivateTransform but is " +
                "%s", private_transform)

        private_transform.set_additional_parameters(
            budget_accountant=self._budget_accountant,
            privacy_id_extractor=self._privacy_id_extractor)
        transformed = self._pcol.pipeline.apply(private_transform, self._pcol)

        return (PrivateCollection(transformed, self._budget_accountant,
                                  self._privacy_id_extractor)
                if private_transform._return_private else transformed)


class MakePrivate(PrivateTransform):
    """Transform class for creating a PrivateCollection."""

    def __init__(self,
                 budget_accountant: budget_accounting.BudgetAccountant,
                 privacy_id_extractor: Callable,
                 label: Optional[str] = None):
        super().__init__(return_private=True, label=label)
        self._budget_accountant = budget_accountant
        self._privacy_id_extractor = privacy_id_extractor

    def expand(self, pcol: pvalue.PCollection):
        return PrivateCollection(pcol, self._budget_accountant,
                                 self._privacy_id_extractor)


class Sum(PrivateTransform):
    """Transform class for performing DP Sum on PrivateCollection."""

    def __init__(self,
                 sum_params: aggregate_params.SumParams,
                 return_private: bool,
                 label: Optional[str] = None):
        super().__init__(return_private=return_private, label=label)
        self._sum_params = sum_params

    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        beam_operations = pipeline_dp.BeamOperations()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant,
                                         beam_operations)

        params = pipeline_dp.AggregateParams(
            noise_kind=self._sum_params.noise_kind,
            metrics=[pipeline_dp.Metrics.SUM],
            max_partitions_contributed=self._sum_params.
            max_partitions_contributed,
            max_contributions_per_partition=self._sum_params.
            max_contributions_per_partition,
            low=self._sum_params.low,
            high=self._sum_params.high,
            public_partitions=self._sum_params.public_partitions)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=self._sum_params.partition_extractor,
            privacy_id_extractor=self._privacy_id_extractor,
            value_extractor=self._sum_params.value_extractor)

        return dp_engine.aggregate(pcol, params, data_extractors)
