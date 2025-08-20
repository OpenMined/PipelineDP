FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git make zsh-theme-powerlevel9k curl \
 && apt-get purge -y --auto-remove \
 && rm -rf /var/lib/apt/lists/*

# For ZSH with powerlevel9k theme
RUN echo "export TERM='xterm-256color'\n \
source /usr/share/powerlevel9k/powerlevel9k.zsh-theme" >> ~/.zshrc

# Set poetry env for docker
ENV YOUR_ENV=${YOUR_ENV} \
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  POETRY_HOME='/usr/local' \
  POETRY_VERSION=2.1.4

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /code

# First copy only the requirements, to cache them in docker layer
COPY poetry.lock pyproject.toml /code/

RUN poetry install --no-interaction --no-ansi --no-root

COPY . /code

# Spark/Beam run on JVM (Java Virtual Machine)
# Therefore, we need to install JRE or JDK to be able to run all the tests
# Here we install OpenJDK-8's JRE
RUN mkdir -p /usr/lib/jvm/java-8-openjdk-amd64/
COPY --from=openjdk:8-jre-slim-bullseye /usr/local/openjdk-8/ /usr/lib/jvm/java-8-openjdk-amd64/

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
