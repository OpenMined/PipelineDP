# PipelineDP prototype

**Warning: this is a prototype not a production implementation.** The main purpose is to illustrate ways how a DP pipeline system might be build. It means:
  1. **Never use is for anonymization of the real data.**
  2. Look at the code with a grain of salt, it's not production ready and likely many parts will be redone in the production implementation. 

It contains computing **count, sum, mean, variance, privacy_id_count per partition key**, it includes:
  1. Performing contribution bounding.
  2. Performing partition selection. 
  3. Aggregations necessary for computing DP metrics (more details in the reference doc).
  4. Adding Laplace noise. 
  5. Running in Apache Beam and w/o any framework (with Python Iterables as input).
  6. Basic privacy budget accounting.
  7. Generating explain calculation reports with information how to DP computations were performed.   


# How to run an example

**aggregate_example.py** contains an example of using the prototype for computing
for each movie in Netflix prize dataset count, unique user count and average
rating.

In order to run an example:
  1. Install Python and run in command line ``pip install numpy apache-beam absl-py``
  2. Download Netflix prize dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data and unpack it.
  3. The dataset itself is pretty big, for speed-up the run it's better to use a part of it, generate a part of it by running in bash:
  
      ```head -10000 combined_data_1.txt > data.txt```
      
      or by other way to get a subset of lines from the dataset.
  4. Run python aggregate_example.py --data_dir=<path to data.txt> --input_file=data.txt  --use_beam=False
  
  On the successful run it should generate file <data_dir>/dp_aggregation_result which should look like
  
  ```(8, MetricsResult(unique_count=14472.412855561333, count=14464.553678887945, sum=None, mean=3.206272313424984, var=None))```
  ```(17, MetricsResult(unique_count=6875.328325189108, count=6802.654799981591, sum=None, mean=2.917327895993826, var=None))```
         
# What should be changed in production implementation.
  1. Unit tests.
  2. Secure noise should be used instead of np noise. 
  3. Implement lazy execution in LocalBackend (which would allow to use more effective privacy budget).
  4. Using class for computing unique privacy units, count, sum, sum^2 instead of vectors.
  5. Using dp_accounting library  https://github.com/google/differential-privacy/tree/main/python/dp_accounting with Privacy Loss Distribution (PLD) for budget accounting (instead of naive composition).

