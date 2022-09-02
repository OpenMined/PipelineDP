import pipeline_dp
from dataclasses import dataclass
import operator


@dataclass
class FrequencyBin:
    lower: int
    count: int
    sum: int
    max: int

    def __add__(self, other: 'FrequencyBin') -> 'FrequencyBin':
        return FrequencyBin(self.lower, self.count + other.count,
                            self.sum + other.sum, max(self.max, other.max))

    def __eq__(self, other):
        return self.lower == other.lower and self.count == other.count and self.sum == other.sum and self.max == other.max


def _to_bin_lower(n: int) -> int:
    bound = 1000
    while n > bound:
        bound *= 10

    round_base = bound // 1000
    return n // round_base * round_base


def _compute_frequency_histogram(
        col, backend: pipeline_dp.pipeline_backend.PipelineBackend):

    col = backend.count_per_element(col, "Frequency of counts")

    # Combiner elements to histogram buckets of increasing sizes. Having buckets
    # of size = 1 is not scalable.
    col = backend.map_tuple(
        col, lambda n, f:
        (_to_bin_lower(n),
         FrequencyBin(lower=_to_bin_lower(n), count=f, sum=f * n, max=n)),
        "To FrequencyBin")

    # (lower_bin_value, FrequencyBin)
    col = backend.combine_per_key(col, operator.add, "Combine FrequencyBins")
    # (lower_bin_value, FrequencyBin)
    col = backend.values(col, "To FrequencyBin")
    # (FrequencyBin,)
    col = backend.to_list(col, "To 1 element collection")

    # 1 element collection: [FrequencyBin]

    def sort_histogram(bins):
        bins.sort(key=lambda bin: bin.lower)
        return bins

    return backend.map(col, sort_histogram, "Sort histogram")
