#Dockerfile
#Copyright 2020 Google LLC. 
#This software is provided as-is, without warranty or representation for any use or purpose. 
#Your use of it is subject to your agreements with Google.
#Base image: https://hub.docker.com/r/frolvlad/alpine-python-machinelearning/dockerfile
#Author: whjang@google.comFROM frolvlad/alpine-python-machinelearningRUN pip3 install — upgrade pandas
RUN pip3 install — upgrade gcsfs
RUN pip3 install — upgrade argparse
ENV PYTHONUNBUFFERED 1RUN mkdir -p /titanic/src
COPY . /titanic/src
WORKDIR /titanic/srcENTRYPOINT [“python”, “titanic_preprocess.py”]