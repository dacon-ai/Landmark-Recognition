FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/dacon-ai/Landmark-Recognition.git

WORKDIR /Landmark-Recognition

RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt