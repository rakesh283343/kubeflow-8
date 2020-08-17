#Dockerfile
#Copyright 2020 Google LLC. 
#This software is provided as-is, without warranty or representation for any use or purpose. 
#Your use of it is subject to your agreements with Google.
#Base image: https://hub.docker.com/r/frolvlad/alpine-python-machinelearning/dockerfile
#Author: whjang@google.com
FROM frolvlad/alpine-python-machinelearning

#RUN pip install --upgrade pandas
RUN pip install --upgrade gcsfs argparse
#RUN pip install --upgrade argparse
ENV PYTHONUNBUFFERED 1

RUN mkdir -p /titanic/src
COPY . /titanic/src
WORKDIR /titanic/src

ENTRYPOINT ["python", "titanic_preprocess.py"]