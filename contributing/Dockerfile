FROM python:3.9.16-slim-bullseye

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
    && rm -rf /var/lib/apt/lists/*