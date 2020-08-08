#Dockerfile
#Copyright 2020 Google LLC. 
#This software is provided as-is, without warranty or representation for any use or purpose. 
#Your use of it is subject to your agreements with Google.
#Base image: https://hub.docker.com/r/frolvlad/alpine-python-machinelearning/dockerfile
#Author: whjang@google.com

#FROM centos:centos7
FROM frolvlad/alpine-python-machinelearning

#RUN yum update -y
#RUN yum install gcc
#RUN pip install -U pip
#RUN yum update
#RUN pip3 install --upgrade virtualenv
#RUN virtualenv linux
#RUN source linux /bin/activate
#RUN python3 -m venv venv
#RUN source venv/bin/activate
#RUN echo pip3 version
# RUN pip3 install --upgrade pycparser
# RUN pip3 install --upgrade cffi==1.14.0
# RUN pip3 install google-crc32c==1.0.0
RUN pip3 install google-cloud-storage==1.0.0
RUN pip3 install --upgrade gcsfs 
RUN pip3 install --upgrade sklearn
RUN pip3 install --upgrade argparse
#RUN pip3 install --upgrade pandas
RUN pip install --upgrade simplejson
ENV PYTHONUNBUFFERED 1
RUN mkdir -p /titanic/src
COPY . /titanic/src
WORKDIR /titanic/src
ENTRYPOINT ["python", "titanic_train.py"]