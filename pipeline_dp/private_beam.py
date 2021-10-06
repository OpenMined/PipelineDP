from apache_beam.transforms import ptransform
from abc import ABC, abstractmethod
from apache_beam.pvalue import PCollection

import pipeline_dp
from pipeline_dp import aggregate_params, budget_accounting


# TODO (prerag) comments and return type
class PrivateCollection(PCollection):

    def __init__(self, pcol, budget_accountant, privacy_id_extractor):
        super(PrivateCollection, self).__init__(pcol.pipeline,
                                                is_bounded=pcol.is_bounded)
        for key, value in pcol.__dict__.items():
            self.__dict__[key] = value
        self.pcol = pcol
        self.budget_accountant = budget_accountant
        self.privacy_id_extractor = privacy_id_extractor

    def __or__(self, private_transform):
        if not isinstance(private_transform, PrivateTransform):
            raise TypeError(
                "private_transform should of type PrivateTransform but is " +
                "%s", private_transform)
        transformed = self.pipeline.apply(private_transform, self)
        return transformed


class PrivateTransform(ptransform.PTransform):

    @abstractmethod
    def expand(self, pcol):
        pass


class MakePrivate(PrivateTransform):

    def __init__(self, budget_accountant, privacy_id_extractor):
        self.budget_accountant = budget_accountant
        self.privacy_id_extractor = privacy_id_extractor

    def expand(self, pcol):
        return PrivateCollection(pcol, self.budget_accountant,
                                 self.privacy_id_extractor)


class Sum(PrivateTransform):

    def __init__(self, sum_params, partition_id_extractor):
        self._sum_params = sum_params
        self._partition_id_extractor = partition_id_extractor

    def expand(self, pcol):
        if not isinstance(pcol, PrivateCollection):
            raise TypeError(
                "pcol should of type PrivateCollection but is " + "%s", pcol)

        beam_operations = pipeline_dp.BeamOperations
        dp_engine = pipeline_dp.DPEngine(pcol.budget_accountant,
                                         beam_operations)

        params = pipeline_dp.AggregateParams(
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
            privacy_id_extractor=pcol.privacy_id_extractor,
            value_extractor=self._sum_params.value_extractor)

        dp_result = dp_engine.aggregate(pcol.pcol, params, data_extractors)

        return self._sum_params
