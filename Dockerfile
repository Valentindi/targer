FROM ubuntu
MAINTAINER Valentin Dittmar mail@valentindittmar.eu

COPY . /targer

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils zsh make g++ wget nano tree curl python3.6 python3.6-dev python3.6-distutils -y

RUN wget https://bootstrap.pypa.io/get-pip.py --no-check-certificate
RUN python3 get-pip.py

RUN python3 -V
RUN pip3 -V
RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools
RUN pip3 install torch numpy scipy allennlp pytorch-pretrained-bert tensorflow
