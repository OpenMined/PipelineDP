# PipelineDP

PipelineDP is a framework for applying differential privacy to large
datasets using batch processing systems such as Apache Spark, Apache Beam,
and more.

To make differential privacy accessible to non-experts, PipelineDP:

* provides a convenient API familiar to Spark or Beam developers
* encapsulates the complexities of differential privacy, such as
* protection of outliers and rare categories, generation of safe
  noise and privacy budget accounting;
* supports many standard computations, such as count, sum, average, and is easily extensible to support other aggregation types.

The project is in the early development stage. More description will be added later.

## Getting started

Take a look on examples of how to run PipelineDP Apache Spark or Beam:

* [Apache Spark example](examples/movie_view_ratings_spark.py)
* [Apache Beam example](examples/movie_view_ratings_beam.py)
* [Local execution example](examples/movie_view_ratings.py)

Here's a code sample for introducing how the private processing code on Spark looks like. For a more complete guide,
please take a look on the examples above, or on the API guide.
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
              low=1,
              high=5,
              # Specifies the aggregation key
              partition_extractor=lambda mv: mv.movie_id,
              # Specifies the value we're aggregating
              value_extractor=lambda mv: mv.rating)
              )
budget_accountant.compute_budgets()

# Save the results
dp_result.saveAsTextFile(FLAGS.output_file)
```

## Development

To install the requirements for local development, run `make dev`.

Please run `make precommit` to auto-format, lint check, and run tests.
Individual targets are `format`, `lint`, `test`, `clean`, `dev`.

### Style guide

Google Python Style Guide https://google.github.io/styleguide/pyguide.html

### Installation

   This project depends on numpy apache-beam pyspark absl-py dataclasses
 
   For installing with poetry please run: 
   
   1. `git clone https://github.com/OpenMined/PipelineDP.git`
   
   2. `cd PipelineDP/`
   
   3. `poetry install `
   

   For installing with pip please run: 
   
   1. `pip install numpy apache-beam pyspark absl-py`
   
   2. (for python 3.6) `pip install dataclasses`
   
### Running end-to-end example
For the development it is convenient to run an end-to-end example. 

For doing this:

1. Download Netflix prize dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data and unpack it.

2. The dataset itself is pretty big, for speed-up the run it's better to use a
part of it. You can generate a part of it by running in bash:

   `head -10000 combined_data_1.txt > data.txt`

   or by other way to get a subset of lines from the dataset.

3. Run python movie_view_ratings.py --input_file=<path to data.txt from 2> --output_file=<...>