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
First refer to the corresponding Jupyter notebook available at
https://github.com/OpenMined/PipelineDP/blob/main/examples/restaurant_visits.ipynb
before running or experimenting with this example.
"""


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


import pipeline_dp
backend = pipeline_dp.LocalBackend()
budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1, total_delta=1e-6)
dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

data_extractors = pipeline_dp.DataExtractors(
   partition_extractor=lambda row: row.day,
   privacy_id_extractor=lambda row: row.visitor_id,
   value_extractor=lambda row: row.spent_money)

params = pipeline_dp.AggregateParams(
   noise_kind=pipeline_dp.NoiseKind.LAPLACE,
   metrics=[pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM],
   max_partitions_contributed=3,
   max_contributions_per_partition=2,
   min_value=0,
   max_value=60)

public_partitions=list(range(1, 8))

dp_result = dp_engine.aggregate(rows, params, data_extractors, public_partitions)
budget_accountant.compute_budgets()
dp_result = list(dp_result)


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
