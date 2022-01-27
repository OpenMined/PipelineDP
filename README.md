# PipelineDP

PipelineDP is a framework for applying differentially private aggregations to large
datasets using batch processing systems such as Apache Spark, Apache Beam,
and more.

To make differential privacy accessible to non-experts, PipelineDP:

* Provides a convenient API familiar to Spark or Beam developers.
* Encapsulates the complexities of differential privacy, such as:
  * protecting outliers and rare categories,
  * generating safe noise,
  * privacy budget accounting.
* Supports many standard computations, such as count, sum, and average. 

Additional information can be found at [pipelinedp.io](https://pipelinedp.io).

## Getting started

Here are some examples of how to use PipelineDP:

* [Apache Spark example](examples/movie_view_ratings/run_on_spark.py)
* [Apache Beam example](examples/movie_view_ratings/run_on_beam.py)
* [Framework-free example](examples/movie_view_ratings/run_without_frameworks.py)
* [Example with all frameworks](examples/movie_view_ratings/run_all_frameworks.py)

Please check out the [codelab](https://github.com/OpenMined/PipelineDP/blob/main/examples/restaurant_visits.ipynb) for a more detailed demonstration of the API functionality and usage.

Code sample showing private processing on Spark:
```python
# Define the privacy budget available for our computation.
budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                      total_delta=1e-6)

# Wrap Spark's RDD into it's private version. You will use this private wrapper
# for all further processing instead of the Spark's RDD. Using the wrapper ensures
# that only private statistics can be released.
private_movie_views = \
    make_private(movie_views, budget_accountant, lambda mv: mv.user_id)

# Calculate the private sum of ratings per movie
dp_result = private_movie_views.sum(
    SumParams(
              # The aggregation key: we're grouping data by movies
              partition_extractor=lambda mv: mv.movie_id,
              # The value we're aggregating: we're summing up ratings
              value_extractor=lambda mv: mv.rating,

              # Limits to how much one user can contribute:
              # .. at most two movies rated per user
              #    (if there's more, randomly choose two)
              max_partitions_contributed=2,
              # .. at most one ratings for each movie
              max_contributions_per_partition=1,
              # .. with minimal rating of "1"
              #    (automatically clip the lesser values to "1")
              min_value=1,
              # .. and maximum rating of "5"
              #    (automatically clip the greater values to "5")
              max_value=5)
              )
budget_accountant.compute_budgets()

# Save the results
dp_result.saveAsTextFile(FLAGS.output_file)
```

## Installation

`pip install pipeline-dp`

Supported Python version >= 3.7.

**Note for Apple Silicon users:** PipelineDP pip package is currently available only 
for x86 architecture. The reason is that [PyDP](https://github.com/OpenMined/PyDP) does not
have pip pacakge. It might be possible to compile it from sources for Apple Silicon.
 
## Development

To install the requirements for local development, run `make dev`.

Please run `make precommit` to auto-format, lint check, and run tests.
Individual targets are `format`, `lint`, `test`, `clean`, `dev`.

### Style guide

We use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

### Running end-to-end example
When developing it is convenient to run an end-to-end example. To do this:

1. Download Netflix prize dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data and unpack it.

2. The dataset itself is pretty big, to speed-up the run it's better to use a
sample of it. Here's how to take a subset of the data in bash:

   `head -10000 combined_data_1.txt > data.txt`

3. Run `python movie_view_ratings.py --input_file=<path to data.txt from 2> --output_file=<...>`
