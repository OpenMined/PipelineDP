"""Report Generator"""

class ReportGenerator:
  """ Generate a report based on the metrics and stages in the pipeline.

  Each ReportGenerator corresponds to one aggregation which contains an
  ordered set of stages. It collects information about the DP aggregation
  and generates a report.
  """

  def __init__(self, params):
    """ Initialize the ReportGenerator.

    Args:
      params (AggregateParams): An object containing a list of parameters.
    """
    self.params = params
    self.stages = []

  def add_stage(self, text):
    """ Add a stage description to the report.

    Args:
     text: string of function that returns string in case of lazy budget split.
    """
    self.stages.append(text)

  def report(self) -> str:
    """ Constructs a report based on stages and metrics.

    Returns:
      A string representation of the constructed report.
    """
    if self.params is not None:
      title = f"Computing metrics: {[m.value[0] for m in self.params.metrics]}"
      result = [f"Differential private: {title}"]
      for i, stage_str in enumerate(self.stages):
        if hasattr(stage_str,
                   "__call__"):
          result.append(f"{i+1}. {stage_str()}")
        else:
          result.append(f"{i+1}. {stage_str}")
      return "\n".join(result)
