def validate_epsilon_delta(epsilon: float, delta: float, obj_name: str):
    """Helper function to validate the epsilon and delta parameters.

  Args:
      epsilon: The epsilon value to validate.
      delta: The delta value to validate.

  Raises:
      A ValueError if either epsilon or delta are out of range.
  """
    if epsilon <= 0:
        raise ValueError(
            f"{obj_name}: epsilon must be positive, not {epsilon}.")
    if delta < 0:
        raise ValueError(
            f"{obj_name}: delta must be non-negative, not {delta}.")
