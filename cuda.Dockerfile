
FROM anibali/pytorch:cuda-9.2

MAINTAINER Valentin Dittmar mail@valentindittmar.eu
USER root
COPY . /targer
WORKDIR /targer
RUN apt-get update
RUN apt-get install make nano wget zip apt-get install gcc-4.9 g++-4.9 -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN ssk

RUN export PYTHONIOENCODING=utf8
RUN LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
RUN export LD_LIBRARY_PATH
RUN pip install -r /targer/requirements.txt
