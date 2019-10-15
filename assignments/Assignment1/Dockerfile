FROM tensorflow/tensorflow:1.14.0-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get install -y git \
                       tzdata &&\
    pip install --upgrade pip &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 

# Install Jupyter lab
WORKDIR /src

RUN pip3 --no-cache-dir install \
         numpy \
         matplotlib \
         scipy \
         scikit-learn \
         seaborn \
         jupyter \
         jupyterlab

# Export port for Jupyter Notebook
EXPOSE 8888
RUN jupyter serverextension enable --py jupyterlab --sys-prefix
WORKDIR /

# By default start bash
ENTRYPOINT ["/bin/bash"]