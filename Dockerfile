FROM python:3.10.13-slim
RUN apt-get update
RUN apt-get -y install \
    wget \ 
    git  \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx

WORKDIR /workspace/