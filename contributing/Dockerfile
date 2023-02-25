FROM python:3.9.16-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git make zsh-theme-powerlevel9k \
 && apt-get purge -y --auto-remove \
 && rm -rf /var/lib/apt/lists/*

# For ZSH with powerlevel9k theme
RUN echo "export TERM='xterm-256color'\n \
source /usr/share/powerlevel9k/powerlevel9k.zsh-theme" >> ~/.zshrc

COPY . /code

WORKDIR /code

RUN pip install --no-cache-dir -r requirements.dev.txt

# Install PipelineDP in Development mode
RUN pip install -e .

# Install process tools
RUN apt-get update && apt-get -y install procps
RUN apt-get -y install software-properties-common
RUN apt-add-repository 'deb http://security.debian.org/debian-security stretch/updates main'

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*