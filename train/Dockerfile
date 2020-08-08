#Dockerfile
#Copyright 2020 Google LLC. 
#This software is provided as-is, without warranty or representation for any use or purpose. 
#Your use of it is subject to your agreements with Google.
#Base image: https://hub.docker.com/r/frolvlad/alpine-python-machinelearning/dockerfile
#Author: whjang@google.com

FROM frolvlad/alpine-python-machinelearning
RUN pip3 install -- upgrade google-cloud-storage
RUN pip3 install -- upgrade gcsfs
RUN pip3 install -- upgrade sklearn
RUN pip3 install -- upgrade argparse
RUN pip3 install -- upgrade pandas
RUN pip3 install -- upgrade simplejson
ENV PYTHONUNBUFFERED 1
RUN mkdir -p /titanic/src
COPY . /titanic/src
WORKDIR /titanic/src
ENTRYPOINT ["python", "titanic_train.py"]