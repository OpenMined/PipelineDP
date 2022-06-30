# Copyright 2022 OpenMined.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This example demonstrates the core functionality of PipelineDP to users who
would like to apply an end-to-end solution to their dataset regardless of their
level of knowledge of differential privacy (DP).

It works on a collection of records that represent restaurant visits for 7 days.
Since a visitor may enter the restaurant multiple times a day as well as several
days a week, each record consists of a visitor_id and day along with
statistically valuable information such as enter_time, spent_minutes
and spent_money.

Differential Privacy offers a tradeoff between the accuracy of aggregations over
statistical databases (such as mean or sum) and the chance of a privacy
violation (such as learning something about individual records in the database).
This tradeoff is an easily configured parameter which can increase privacy by
decreasing accuracy of aggregations or vice versa.
While other anonymization schemes (such as k-anonymity) completely fail on
increases in data release, differential privacy degrades slowly.

Please refer to the blog post available at
https://desfontain.es/privacy/friendly-intro-to-differential-privacy.html
for a high-level, non-technical introduction to differential privacy and
the book "The Algorithmic Foundations of Differential Privacy" available at
https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
for a more detailed explanation.
"""


"""
Some terminology for the uninitiated:
(those with knowledge can skip this section)

1) (Privacy) budget:
Every operation leaks some information about individuals.
The total privacy cost of a pipeline is the sum of the costs of its releases.
This cost is supposed to be below a certain total cost known as the privacy budget.
Typically, the greek letters 'epsilon' and 'delta' (ϵ and δ) are used to define the budget.

2) Partition:
A partition is a subset of the data corresponding to a given value of the
aggregation criterion. In this example, the partitions are the seven days of the week.

3) Partition key:
This is the partition identifier.
In this example, the partition key is a day as the data are aggregated per day.

4) Privacy ID:
An ID of the unit of privacy under DP protection.
In this example, since the visitors are being protected, the privacy ID is the visitor_id.

5) PipelineDP:
PipelineDP is an end-to-end Python framework for generating differentially private statistics.
It provides a high-level API for anonymizing data either locally or using
various supported pipeline frameworks such as Apache Beam.
This example demonstrates the core API on a local dataset.
"""


# The following code aggregates number of visits on a day (count aggregation)
# and total money spent per day (sum aggregation) in the usual non-DP manner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('restaraunts_week_data.csv')
df.rename(inplace=True,
          columns={'VisitorId' : 'visitor_id',
                   'Time entered' : 'enter_time',
                   'Time spent (minutes)' : 'spent_minutes',
                   'Money spent (euros)' : 'spent_money',
                   'Day' : 'day'})
rows = [index_row[1] for index_row in df.iterrows()]

non_dp_count = [0] * 7
non_dp_sum = [0] * 7
for row in rows:
  index = row['day'] - 1
  non_dp_count[index] += 1
  non_dp_sum[index] += row['spent_money']


# Now, to calculate differentially private aggregations using PipelineDP, first import the library
import pipeline_dp


# Select the framework/backend
# In this case the local backend as this example demonstrates the core API on a local dataset.
backend = pipeline_dp.LocalBackend()


"""
Configure privacy budget (BudgetAccountant)

BudgetAccountant defines the total amount of privacy budget that will be spent
on DP aggregations within the program.

It automatically splits the budget among all DP aggregations.

Depending on the type of BudgetAccountant different budget allocation strategies
can be applied.

This example uses the NaiveBudgetAccountant which implements basic composition.
"""
budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1, total_delta=1e-6)


# Instantiate the DPEngine which performs DP operations.
dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)


"""
Configure data extractors

Instruct PipelineDP on how to extract the privacy ID, the partition key and the
value to be aggregated from the input data.

This is achieved by configuring DataExtractors as shown in the code below.

The extractor functions will be applied to each row of the input data.
"""
data_extractors = pipeline_dp.DataExtractors(
   partition_extractor=lambda row: row.day,
   privacy_id_extractor=lambda row: row.visitor_id,
   value_extractor=lambda row: row.spent_money)


"""
Configure DP aggregations

This is achieved by defining an AggregationParams object with the following properties:

    1) noise_kind defines the distribution of the noise that is added to make the result differentially private.

    2) methods collection defines aggregation methods that will be executed on the dataset.
       In this example visits are counted and visitor spendings are summed.
       Both are executed over the partition 'day' in this example.

    3) max_partitions_contributed specifies the upper bound on the number of partitions
       to which one privacy ID (visitor_id in this example) can contribute.
       All contributions in excess of the limit will be discarded.
       The contributions to be discarded are chosen randomly.

    4) max_contributions_per_partition is the maximum number of times a privacy ID can contribute to a partition.
       For instance, if in this example it's set to 2, it means that for each
       visitor PipelineDP will count up to 2 visits and corresponding spendings per day.

    5) min_value and max_value are the lower and upper bounds on the privacy ID contributions.
       This is necessary to limit sensitivity from individual contributions.
       Values less than min_value are "clamped" to the min_value
       and values greater than max_value are "clamped" to the max_value.
"""
params = pipeline_dp.AggregateParams(
   noise_kind=pipeline_dp.NoiseKind.LAPLACE,
   metrics=[pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM],
   max_partitions_contributed=3,
   max_contributions_per_partition=2,
   min_value=0,
   max_value=60)


"""
Configure public_partitions

This is an optional parameter to the aggregate method of a DPEngine instance.
By default it is set to "None" and PipelineDP will select partitions to release
in a DP manner. If specified, it has to be a collection of partition keys that
will appear in the result.

In this example, since statistics are desired for all days of a week the
public partitions would be the range(1, 8); however, it can be set to
range(6, 8) for weekends-only DP computations.
"""
public_partitions=list(range(1, 8))


"""
Run the pipeline

Now, that all the parameters have been defined, call aggregate on the DPEngine
instance. This is a lazy operation, it builds the computational graph but
doesn't trigger any data processing.

Next, call budget_accountant.compute_budgets() so that it allocates a privacy
budget to the aggregations.

Finally, trigger the pipeline computation and obtain the result.

Please note that the BudgetAccountant is stateful implying that the code below
can be executed only once. New BudgetAccountant and DPEngine instances are
required for recomputation of DP result - which means the privacy budget is
consumed on every pipeline run.
"""
# Build computational graph for aggregation
dp_result = dp_engine.aggregate(rows, params, data_extractors, public_partitions)

# Compute budget per each DP operation. 
budget_accountant.compute_budgets()

# Run computation.
dp_result = list(dp_result)


"""
Display DP results as compared to Non-Dp results computed above.

The result includes 7 partitions: 1 partition per day, and 2 values associated
with each partition: the count of visits and total spending.

In this example public_partitions parameter was specified - i.e. the partition
keys that appear in the result were defined explicitly. It's possible in this
case because the week days are publicly known information. However, defining
public_partitions isn't always possible - if, for example, partition keys are
based on user data rather than being public information, they are private
information and need to be calculated using DP.

If public_partitions isn't specified, PipelineDP automatically calculates
partition keys with DP. As a consequence, the DP result will include only
partitions that have sufficiently many contributing privacy IDs to ensure that a
single privacy ID cannot impact the structure of the returned result.

See the blog post about private partition selection available at
https://desfontain.es/privacy/almost-differential-privacy.html.

Further, public_partitions has a couple of caveats.
First, one must be absolutely sure that the provided partitions are either based
on public knowledge or derived using differential privacy.
Second, partitions with no contributions from users will appear in the
DP statistics with noisy values. This ensures that an attacker cannot know which
partitions users contributed to by looking at the structure of the results;
however, this can be detrimental for utility, as empty partitions will be all noise and no signal.
"""
dp_sum = [0] * 7
dp_count = [0] * 7
for count_sum_per_day in dp_result:
  index =  count_sum_per_day[0] - 1
  dp_count[index] = count_sum_per_day[1][0]
  dp_sum[index] = count_sum_per_day[1][1]

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
x = np.arange(len(days))

width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, non_dp_count, width, label='non-DP')
rects2 = ax.bar(x + width/2, dp_count, width, label='DP')
ax.set_ylabel('Visit count')
ax.set_title('Count visits per day')
ax.set_xticks(x)
ax.set_xticklabels(days)
ax.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, non_dp_sum, width, label='non-DP')
rects2 = ax.bar(x + width/2, dp_sum, width, label='DP')
ax.set_ylabel('Sum spendings')
ax.set_title('Money spent per day')
ax.set_xticks(x)
ax.set_xticklabels(days)
ax.legend()
fig.tight_layout()
plt.show()
