# Copyright 2022 OpenMined.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# base image 
FROM ubuntu:20.04

# maintainer
MAINTAINER wenhui zhang<wenhuizhang.psu@gmail.com> <wenhui.zhang@bytedance.com> <wenhui@gwmail.gwu.edu>

# commands executed in terminal
RUN set -x

# build deps
RUN apt-get update
RUN apt-get -y install git vim  
RUN apt-get -y install libssl-dev 
RUN apt-get -y install ca-certificates
RUN apt-get -y install curl
RUN apt-get -y install gnupg
RUN apt-get -y install lsb-release 
#RUN apt-get -y install default-jre
RUN apt-get -y install python3 python3-pip

RUN pip3 install pandas pipeline-dp apache-beam absl-py


# download source 
RUN git clone https://github.com/OpenMined/PipelineDP.git
RUN cd /PipelineDP && git checkout v0.1.1

# check modes and run example 

RUN python3 /PipelineDP/examples/restaraunt_visits/run_without_frameworks.py --input_file=/PipelineDP/examples/restaraunt_visits/restaraunts_week_d
RUN cat /PipelineDP/examples/restaraunt_visits/out.txt
