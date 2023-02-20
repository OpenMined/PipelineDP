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