---
layout: page
title: Key definitions
description: >-
  Below we summarize the key concepts that are used in the code and in the
  documentation.
---

A simplified visual summary of some key concepts we use in PipelineDP:

![PipelineDP terminology]({{ "/assets/images/terminology.png" | relative_url }})

A *record* is an element in the input dataset.

A *partition* is a subset of the data corresponding to a given value of the
aggregation criterion. Usually we want to aggregate each partition separately.
For example, if we count visits to restaurants, the visits for one particular
restaurant are a single partition, and the count of visits to that restaurant
would be the aggregate for that partition.

A *partition key* is the aggregation key corresponding to a partition.

A *privacy unit* is an entity that we’re trying to protect with differential
privacy. Often, this refers to a single individual. An example of a more complex
privacy unit is a person+restaurant pair, which protects all visits by an
individual to a particular restaurant or, in other words, the fact that a
particular person visited any particular restaurant.

A *privacy ID* is an identifier of a privacy unit.

*Contribution bounding* is a process of limiting contributions by a single
individual (or an entity represented by a privacy key) to the output dataset or
its partition. This is key for DP algorithms, since protecting unbounded
contributions would require adding infinite noise.

*Cross-partition contribution bounding* is a procedure which ensures that each
individual contributes to a limited number of partitions.

*Per-partition contribution bounding* is a procedure which ensures that each
individual’s contribution to any single partition is bounded.

*Partition selection* is a process of identifying the partition keys that are
safe to release in the sense that they don't break the DP guarantees and don't
leak any user information.

*Public partitions* are partition keys that are publicly known and hence don't
leak any user information. An example of public partitions could be week days.

*(Privacy) budget*: every operation leaks some information about individuals.
The total privacy cost of a pipeline is the sum of the costs of its releases.
You want this to be below a certain total cost. That's your budget. Typically,
the greek letters 'epsilon' and 'delta' (ϵ and δ) are used to define the budget.

<a class="c-button c-button--primary" href="https://pipelinedp.io/overview/" target="_blank">Overview</a>
