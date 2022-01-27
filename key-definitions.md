---
layout: page
title: Key definitions
description: >-
Below we summarize the key concepts that are used in the code and in the
documentation.
---
A *partition* is a subset of the data corresponding to a given value of the aggregation criterion. Usually we want to aggregate each partition separately. For example, if we count visits to restaurants, the visits for one particular restaurant are a single partition, and the count of visits to that restaurant would be the aggregate for that partition.

A *partition key* is the aggregation key corresponding to a partition.

A *privacy unit* is an entity that we’re trying to protect with differential privacy. Often, this refers to a single individual. An example of a more complex privacy unit is a person+restaurant pair, which protects all visits by an individual to a particular restaurant or, in other words, the fact that a particular person visited any particular restaurant.

A *privacy ID* is an identifier of a privacy unit.

*Contribution bounding* is a process of limiting contributions by a single individual (or an entity represented by a privacy key) to the output dataset or its partition. This is key for DP algorithms, since protecting unbounded contributions would require adding infinite noise.

*Partition selection* is a process of identifying the partition keys that are
safe to release in the sense that they don't break the DP guarantees and don't
leak any user information.