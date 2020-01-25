FROM nvidia/cuda:9.0-base

MAINTAINER Valentin Dittmar mail@valentindittmar.eu

COPY . /targer
RUN add-apt-repository -r ppa:jonathonf/python-3.6 -y
#RUN apt-key update -y
RUN apt update
#RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends apt-utils zsh make wget nano curl unzip  g++ -y 

RUN wget https://bootstrap.pypa.io/get-pip.py --no-check-certificate
RUN python3 get-pip.py

RUN python3 -V
RUN pip3 -V
RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools
RUN pip3 install torch numpy scipy allennlp pytorch-pretrained-bert tensorflow
