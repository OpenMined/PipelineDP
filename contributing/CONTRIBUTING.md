# Development with Docker

To make it easier to contribute to PipelineDP, we've included a Dockerfile that can be used to run PipelineDP inside a Docker container.

First, fork the repo to your github account.

Then, clone the repo on your computer:
```
git clone https://github.com/HERE-IS-YOUR-ID/PipelineDP
```

Go to the project folder:
```
cd PipelineDP
```

Build the image using the `Dockerfile` provided in the `contributing` folder:
```
docker build -t devpipelinedp:latest -f contributing/Dockerfile . 
```

Now every time you want to develop, you have to:

1. Open the project folder (PipelineDP) in your favorite IDE as usual
2. Open the terminal and run:
```
(Windows): docker run -it --rm -v ${pwd}:/code -w /code devpipelinedp:latest zsh
(Unix): docker run -it --rm -v $(pwd):/code -w /code devpipelinedp:latest zsh
```
3. Edit your files in your IDE as usual: **the local changes will be reflected in the container**
4. Run commands in the terminal opened, to have them run in the container

Note that the `docker run` command above:
- Runs a container with a mapped volume with read/write permission on the project folder: so it can access the local/host project files and have the local changes reflected in the container.
- Removes the container every time you close the terminal (which is fine, since the changes remain in the local files)


# Running end-to-end example

When developing it is convenient to run an end-to-end example. 

The example below uses a sample from the `combined_data_1.txt` in the [Netflix prize dataset](https://www.kaggle.com/netflix-inc/netflix-prize-data).

To run the example, in the terminal:
```
python -m examples.movie_view_ratings.run_all_frameworks \
       --input_file=contributing/sample_combined_data_1.txt \
       --output_file=result_example.txt
```

And see the results by looking at the file `result_example.txt`.


# Style guide
We use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).


# Submitting changes

Before submitting your changes, please make sure to auto-format, lint check, and run tests, by running:
```
make precommit
```

Individual targets are format, lint, test, clean, dev.
