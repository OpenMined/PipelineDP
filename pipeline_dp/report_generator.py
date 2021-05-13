"""Generator for explain DP computations reports."""
from pipeline_dp.aggregate_params import AggregateParams
class ReportGenerator:
  """Generate a report based on the metrics and stages in the pipeline.

  Each ReportGenerator corresponds to one aggregation which contains an
  ordered set of stages. It collects information about the DP aggregation
  and generates a report.
  """

  def __init__(self, params: AggregateParams):
    """Initialize the ReportGenerator.

    Args:
      params (AggregateParams): An object containing a list of parameters.
    """
    self._params = params
    self._stages = []

  def add_stage(self, text: str):
    """Add a stage description to the report.

    Args:
     text: string of function that returns string in case of lazy budget split.
    """
    self._stages.append(text)

  def report(self) -> str:
    """Constructs a report based on stages and metrics.

    Returns:
      A string representation of the constructed report.
    """
    if self._params is not None:
      title = f"Computing metrics: {[m.value[0] for m in self._params.metrics]}"
      result = [f"Differential private: {title}"]
      for i, stage_str in enumerate(self._stages):
        if hasattr(stage_str, "__call__"):
          result.append(f"{i+1}. {stage_str()}")
        else:
          result.append(f"{i+1}. {stage_str}")
      return "\n".join(result)
