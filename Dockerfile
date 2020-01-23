FROM python:3.5-alpine
MAINTAINER Valentin Dittmar mail@valentindittmar.eu

COPY . /targer

RUN apk add zsh tree curl wget
RUN apk --no-cache --update-cache add gcc gfortran python python-dev py-pip build-base wget freetype-dev libpng-dev openblas-dev
RUN pip install --upgrade pip
RUN pip install -U setuptools

RUN pip install torch numpy scipy allennlp pytorch-pretrained-bert tensorflow
