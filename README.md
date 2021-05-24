# PipelineDP

PipelineDP is project for performing Differentially Private (DP) aggregations in Python Data Pipelines.

The project is in the early development stage. More description will be added later.

## Development

### Style guide

Google Python Style Guide https://google.github.io/styleguide/pyguide.html

### Installation

   This project depends on numpy apache-beam pyspark absl-py dataclasses
 
   For installing with poetry please run: 
   
   1. `git clone https://github.com/OpenMined/PipelineDP.git`
   
   2. `cd PipelineDP/`
   
   3. `poetry install `
     
   
### Running end-to-end example
For the development it is convenient to run an end-to-end example. 

For doing this:

1.Download Netflix prize dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data and unpack it.

2.The dataset itself is pretty big, for speed-up the run it's better to use a
part of it. You can generate a part of it by running in bash:

   `head -10000 combined_data_1.txt > data.txt`

   or by other way to get a subset of lines from the dataset.

1. Run python movie_view_ratings.py --input_file=<path to data.txt from 2> --output_file=<...>