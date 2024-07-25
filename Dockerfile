# base image with cuda 12.1 and pytorch 2.2.0
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

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

# copy conda.yaml file into the Docker image
COPY configs/conda.yaml .

# create the Conda environment
RUN /root/miniconda3/bin/conda create -n gamba_az python=3.12.2

# install all needed for mamba
RUN /root/miniconda3/bin/conda install -n gamba_az -c nvidia cuda-nvcc=12.1 && \
  /root/miniconda3/bin/conda run -n gamba_az pip install torch==2.2.0 && \
  /root/miniconda3/bin/conda run -n gamba_az pip install packaging==23.2 && \
  /root/miniconda3/bin/conda run -n gamba_az pip install causal-conv1d==1.3.0.post1 && \
  /root/miniconda3/bin/conda run -n gamba_az pip install mamba-ssm==2.1.0 && \
  /root/miniconda3/bin/conda run -n gamba_az pip install flash_attn==2.5.9.post1 && \
  /root/miniconda3/bin/conda run -n gamba_az conda install conda-forge::threadpoolctl && \
  /root/miniconda3/bin/conda install -n gamba_az -c defaults mkl && \ 
  /root/miniconda3/bin/conda run -n gamba_az pip install wandb==0.16.5 && \
  /root/miniconda3/bin/conda run -n gamba_az conda env update -f conda.yaml

# activate the environment and ensure nvcc is available
SHELL ["/root/miniconda3/bin/conda", "run", "-n", "gamba_az", "/bin/bash", "-c"]

# default command to run when the container starts
CMD ["bash"]