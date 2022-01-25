# PipelineDP

PipelineDP is a framework for applying differentially private aggregations to large
datasets using batch processing systems such as Apache Spark, Apache Beam,
and more.

To make differential privacy accessible to non-experts, PipelineDP:

* provides a convenient API familiar to Spark or Beam developers.
* encapsulates the complexities of differential privacy, such as:
  * protecting outliers and rare categories
  * generating safe noise
  * privacy budget accounting
* supports many standard computations, such as count, sum, and average, and is easily extensible to support other aggregation types.

The project is in the early development stage. More description will be added later.

## Getting started

Take a look at examples of how to run PipelineDP:

* [Apache Spark example](examples/movie_view_ratings_spark.py)
* [Apache Beam example](examples/movie_view_ratings_beam.py)
* [Framework-free example](examples/movie_view_ratings_local.py)
* [Example with all frameworks](examples/movie_view_ratings.py)

For a deep dive on the API take a look at the [codelab](https://github.com/OpenMined/PipelineDP/blob/main/examples/restaurant_visits.ipynb).

Here's a code sample that introduces private processing on Spark. For a more complete guide,
please take a look on the examples above, or the API guide.
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
    SumParams(max_partitions_contributed=2,
              max_contributions_per_partition=2,
              min_value=1,
              max_value=5,
              # Specifies the aggregation key
              partition_extractor=lambda mv: mv.movie_id,
              # Specifies the value we're aggregating
              value_extractor=lambda mv: mv.rating)
              )
budget_accountant.compute_budgets()

# Save the results
dp_result.saveAsTextFile(FLAGS.output_file)
```

For more documentation check [pipelinedp.io](https://pipelinedp.io).
## Installation

`pip install pipeline-dp`

Supported Python version >= 3.7.

## Development

To install the requirements for local development, run `make dev`.

Please run `make precommit` to auto-format, lint check, and run tests.
Individual targets are `format`, `lint`, `test`, `clean`, `dev`.

### Style guide

We use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

   
### Running end-to-end example
When developing it is convenient to run an end-to-end example. 

To do this:

1. Download Netflix prize dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data and unpack it.

2. The dataset itself is pretty big, to speed-up the run it's better to use a
sample of it. Here's how to take a subset of the data in bash:

   `head -10000 combined_data_1.txt > data.txt`

3. Run python movie_view_ratings.py --input_file=<path to data.txt from 2> --output_file=<...>
