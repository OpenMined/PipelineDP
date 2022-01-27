---
layout: page
title: Get started
description: >-
  Gain a high-level understanding of how PipelineDP works as well as some of the
  foundational design decisions behind the project.
---

# Setting up your environment
Here’s how you set up PipelineDP on your computer:

```python
# Check that your Python version is 3.7 or greater
python --version

python -m venv demo-pipelinedp

. /demo-pipelinedp/bin/activate

# Install PipelineDP
pip install pipeline-dp
```

# Trying it out
### Quick tour (5 min, no setup needed)

A simple example that shows how to calculate restaurant visits with differential privacy. 

<a class="c-hero__button c-button c-button--secondary" style="margin: 0px;" href="https://github.com/OpenMined/PipelineDP/blob/main/examples/quickstart.ipynb" target="_blank">View as Jupiter Notebook</a>
<a class="c-hero__button c-button c-button--primary" style="margin: 0px;" href="https://colab.research.google.com/github/OpenMined/PipelineDP/blob/main/examples/quickstart.ipynb" target="_blank">Run in Google Colab</a>

### Advanced tour (1 hour, no setup needed)
A deeper walk-through: learn the key concepts of differential privacy and PipelineDP API. 

<a class="c-hero__button c-button c-button--secondary" style="margin: 0px;" href="https://github.com/OpenMined/PipelineDP/blob/main/examples/restaurant_visits.ipynb" target="_blank">View as Jupiter Notebook</a>
<a class="c-hero__button c-button c-button--primary" style="margin: 0px;" href="https://colab.research.google.com/github/OpenMined/PipelineDP/blob/main/examples/restaurant_visits.ipynb" target="_blank">Run in Google Colab</a>

### Run an example locally (15 min, requires setting up Python environment)
If you’d like to plan to run an example on your computer instead of Jupiter notebook, please go through the “Setting up the environment” section below and run:

```python
# 1. Follow the “set up the environment” section to install PipelineDP
# ...

# 2. Download and execute example code from git
git clone https://github.com/OpenMined/PipelineDP.git
cd PipelineDP/examples/restaraunt_visits/
pip install pandas absl-py
python run_without_frameworks.py --output_file=output.txt

# 3. Check the results 
cat output.txt

# 4. Look inside run_without_frameworks.py file, play with parameters and metrics
```


