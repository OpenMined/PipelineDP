---
layout: page
title: Get started
description: >-
  Get up and running with PipelineDP by learning how to set up your environment
  and start running examples locally.
---

# PipelineDP overview

PipelineDP is a Python open source framework for applying differentially private aggregations to 
large datasets using batch processing systems such as Apache Spark, Apache Beam, and more.

To make differential privacy accessible to non-experts, PipelineDP:

* Provides a convenient API familiar to Spark or Beam developers.
* Encapsulates the complexities of differential privacy, such as:
  * protecting outliers and rare categories,
  * generating safe noise,
  * privacy budget accounting.
* Supports standard computations: count, sum, and average (and soon more).

*Note* that this project is still experimental and is subject to change.
We discourage its usage in production systems as it's not
tested thoroughly yet and might potentially break. You can learn more in the
[Roadmap section](https://pipelinedp.io/overview/#roadmap).

# Setting up your environment
Here’s how you set up PipelineDP on your computer:

{% highlight shell_session %}
# Check that your Python version is 3.7 or greater
$ python --version

# Create and activate a Python virtual environment
$ python -m venv demo-pipelinedp
$ source demo-pipelinedp/bin/activate

# Install PipelineDP
$ pip install pipeline-dp
{% endhighlight %}

# Trying it out
### Quick tour (5 min, no setup needed)

A simple example that shows how to calculate restaurant visits with differential privacy. 

<a class="c-hero__button c-button c-button--secondary" style="margin: 0px;" href="https://github.com/{{ site.github_repository }}/blob/main/examples/quickstart.ipynb" target="_blank">View as Jupiter Notebook</a>
<a class="c-hero__button c-button c-button--primary" style="margin: 0px;" href="https://colab.research.google.com/github/{{ site.github_repository }}/blob/main/examples/quickstart.ipynb" target="_blank">Run in Google Colab</a>

### Advanced tour (1 hour, no setup needed)
A deeper walk-through: learn the key concepts of differential privacy and PipelineDP API. 

<a class="c-hero__button c-button c-button--secondary" style="margin: 0px;" href="https://github.com/{{ site.github_repository }}/blob/main/examples/restaurant_visits.ipynb" target="_blank">View as Jupiter Notebook</a>
<a class="c-hero__button c-button c-button--primary" style="margin: 0px;" href="https://colab.research.google.com/github/{{ site.github_repository }}/blob/main/examples/restaurant_visits.ipynb" target="_blank">Run in Google Colab</a>

### Run an example locally (15 min, requires setting up Python environment)
If you’d like to plan to run an example on your computer instead of Jupiter notebook, please go through the “Setting up the environment” section below and run:

{% highlight shell_session %}
# 1. Follow the “set up the environment” section above to install PipelineDP

# 2. Download and execute example code from git
$ git clone https://github.com/{{ site.github_repository }}.git
$ cd PipelineDP/examples/restaraunt_visits/
$ pip install pandas absl-py
$ python run_without_frameworks.py --output_file=output.txt

# 3. Check the results 
$ cat output.txt

# 4. Look inside run_without_frameworks.py file, play with parameters and metrics
{% endhighlight %}
