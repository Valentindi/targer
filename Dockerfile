FROM ubuntu
MAINTAINER Valentin Dittmar mail@valentindittmar.eu

COPY . /targer

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils zsh nano tree curl python3.6 python3.6-dev python3.6-distutils -y

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.6

RUN python -V
RUN pip -v
RUN pip install --upgrade pip
RUN pip install -U setuptools
RUN pip install torch numpy scipy allennlp pytorch-pretrained-bert tensorflow
