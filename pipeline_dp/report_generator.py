"""Generator for explaining DP computation reports."""

from pipeline_dp.aggregate_params import AggregateParams


class ReportGenerator:
    """Generates a report based on the metrics and stages in the pipeline.

  Each ReportGenerator corresponds to one aggregation which contains an
  ordered set of stages. It collects information about the DP aggregation
  and generates a report.
  """

    def __init__(self, params):
        """Initialize the ReportGenerator."""
        self._params_str = None
        if params:
            self._params_str = str(params)
        self._stages = []

    def add_stage(self, text: str):
        """Add a stage description to the report."""
        self._stages.append(text)

    def report(self) -> str:
        """Constructs a report based on stages and metrics."""
        if not self._params_str:
            return ""
        title = f"Computing <{self._params_str}>"
        result = [f"Differentially private: {title}"]
        for i, stage_str in enumerate(self._stages):
            if hasattr(stage_str, "__call__"):
                result.append(f"{i+1}. {stage_str()}")
            else:
                result.append(f"{i+1}. {stage_str}")
        return "\n".join(result)
