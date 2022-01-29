# PipelineDP Website

This is the source code for the [pipelinedp.io](https://pipelinedp.io/) website,
powered by [Jekyll](https://jekyllrb.com/).

## Contributing

Before you begin, you'll need to have [Ruby](https://www.ruby-lang.org/) and
[Bundler](https://bundler.io/) installed. You can find the installation
instructions on their websites.

Install the necessary project dependencies by running:

```bash
bundle install
```

To start a local development server at
[http://localhost:4000](http://localhost:4000), run the following command from
the root of the project:

```bash
bundle exec jekyll serve
```

To build the site to the `_site/` directory, run the following command from the
root of the project:

```bash
bundle exec jekyll build
```

*Note: You should prefix all jekyll commands with `bundle exec` to make sure
you're using the jekyll version defined in the `Gemfile`.*
