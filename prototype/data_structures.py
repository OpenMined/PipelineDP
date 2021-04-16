from enum import Enum
from typing import List, Optional, Union, Iterable, Any

from dataclasses import dataclass
import numpy as np


@dataclass
class MetricsResult:
  privacy_id_count: Optional[int] = None
  count: Optional[int] = None
  sum: Optional[float] = None
  mean: Optional[float] = None
  var: Optional[float] = None


class NoiseKind(Enum):
  LAPLACIAN = 'laplacian'
  GAUSSIAN = 'gaussian'  # not supported yet


class Metrics(Enum):
  COUNT = 'count',
  PRIVACY_ID_COUNT = 'privacy_id_count',
  SUM = 'sum',
  MEAN = 'mean',
  STD = 'std',
  VAR = 'variance'
  SUM2 = 'sum of squares'


@dataclass
class AggregateParams:
  max_partitions_contributed: int
  max_contributions_per_partition: int
  low: float
  high: float
  metrics: Iterable[Metrics]
  budget_weight: float = 1
  preagg_partition_selection: bool = True
  public_partitions: Any = None