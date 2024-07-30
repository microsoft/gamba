# base image with cuda 12.1 and pytorch 2.2.0
FROM ptebic.azurecr.io/public/aifx/acpt/stable-ubuntu2004-cu121-py310-torch221:biweekly.202403.1.v1

# need wget and gawk
RUN apt-get update && apt-get install -y wget gawk

# miniconda
RUN mkdir -p /root/miniconda3 && \
    wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm -rf /root/miniconda3/miniconda.sh && \
    /root/miniconda3/bin/conda init bash && \
    /root/miniconda3/bin/conda init zsh

# git
RUN apt-get update && apt-get install -y git

# environment variables
ENV PATH=/root/miniconda3/bin:$PATH

# working directory
WORKDIR /gamba

# install everything needed for mamba
RUN conda install python=3.12.2 && \
pip install nvidia-cuda-nvcc-cu12 && \
  pip install torch==2.2.0 && \
  pip install packaging==23.2 && \
  pip install causal-conv1d==1.3.0.post1 && \
  pip install mamba-ssm==2.1.0 && \
  pip install flash_attn==2.5.9.post1 && \
  conda install conda-forge::threadpoolctl && \
  pip install defaults mkl && \ 
  pip install sequence-models &&\
  pip install evodiff && \
  pip install wandb==0.16.5 


# default command to run when the container starts
CMD ["bash"]