
FROM anibali/pytorch:cuda-9.2

MAINTAINER Valentin Dittmar mail@valentindittmar.eu
USER root
COPY . /targer
WORKDIR /targer
RUN apt-get update
RUN apt-get install make nano wget zip apt-get install gcc-4.9 g++-4.9 -y
RUN export PYTHONIOENCODING=utf8
RUN LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
RUN export LD_LIBRARY_PATH
RUN pip install -r /targer/requirements.txt
