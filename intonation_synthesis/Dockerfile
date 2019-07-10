FROM nvidia/cuda:9.0-base-ubuntu16.04

# ENV TRAIN_ON_GPU True

RUN apt-get -y update
RUN apt-get install -y --fix-missing --no-install-recommends \
    build-essential \
    checkinstall \
    make \
    git \
    nginx \
    ca-certificates \
    python3 \
    python3-pip \
    python3-setuptools \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    cuda9.2 \
    cuda-toolkit-9-2 \
    cuda-visual-tools-9-2 \
    cuda-cusparse-dev-9-2 \
    cuda-cusolver-dev-9-2 \
    cuda-curand-dev-9-2 \
    cuda-cufft-dev-9-2 \
    cuda-cudart-dev-9-2 \
    cuda-cublas-dev-9-2 \
    cuda-command-line-tools-9-2 \
    libcudnn7 \
    libcudnn7-dev

# set working directory
RUN mkdir -p /app
WORKDIR /app

RUN mkdir -p /app/entrypoints

ADD __init__.py .
ADD datasets.py .
ADD net.py .
ADD train.py .
ADD serve.py .
ADD wsgi.py .
ADD nginx.conf .
ADD predictor.py .
ADD utils.py .
ADD const.py .
ADD settings_prod.py ./settings.py

ADD requirements.txt .
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt

RUN git clone https://github.com/CSTR-Edinburgh/merlin.git && mv merlin merlin_repo && ln -s merlin_repo/src/ merlin && cd merlin_repo && pip install -r requirements.txt

ADD train.sh /app/entrypoints/train
ADD serve.sh /app/entrypoints/serve
RUN chmod +x /app/entrypoints/train /app/entrypoints/serve
ENV PATH=/app/entrypoints:${PATH}

ENTRYPOINT ["python3","train.py"]
