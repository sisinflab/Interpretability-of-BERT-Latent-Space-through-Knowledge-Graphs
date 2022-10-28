FROM tensorflow/tensorflow:latest-gpu

ENV DEBIAN_FRONTEND noninteractive

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.9 python3-pip python3-setuptools python3.9-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN python3.9 -m pip install -r requirements.txt
RUN python3.9 -m pip install --upgrade protobuf==3.20.0

COPY . .

RUN python3.9 -m pip install -e .

RUN ln -sf /usr/bin/python3.9 /usr/bin/python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3
