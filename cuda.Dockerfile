FROM anibali/pytorch:cuda-9.2

MAINTAINER Valentin Dittmar mail@valentindittmar.eu
USER root
COPY . /targer
WORKDIR /targer


RUN pip3 install -r /targer/requirements.txt
