FROM ubuntu
MAINTAINER Valentin Dittmar mail@valentindittmar.eu

COPY . /targer

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install zsh -y
RUN apt-get install nano -y
RUN apt-get install tree -y
RUN apt-get install wget -y
RUN apt-get install unzip -y
# Python package management and basic dependencies
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils -y

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

#RUN apt-get install software-properties-common -y
#RUN add-apt-repository ppa:deadsnakes/ppa -y
#RUN apt-get update -y
#RUN apt-get install python3.7 -y

RUN python -V
RUN pip -v
RUN python -m pip install --upgrade pip
RUN pip install -r /targer/requirements.txt

