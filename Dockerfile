FROM tensorflow/tensorflow:latest-gpu-py3

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