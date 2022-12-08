# use this image https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
# update ubuntu
RUN apt-get -y update
# install git 
RUN apt-get -y install git

# create a working directory 
# This instructs Docker to use this path as the default location 
# for all subsequent commands
# you can call it whatever you want 
WORKDIR /InnoLab
RUN apt-get update && \
    # -y means --assume-yes 
    apt-get install --no-install-recommends -y \
    # build-essential is packages for gcc 
    build-essential \
    # install python3 and pip 
    python3 python3-pip && \
    # clean archive files 
    apt-get clean && rm -rf /var/lib/apt/lists/*

# update pip 
RUN pip3 install --upgrade pip

# install JAX seperatly 
RUN pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install pytorch 
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# copy requirements
COPY requirements.txt requirements.txt
# run pip3 install without cache 
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
# add source code (all things we installed into the base image)
# COPY . .
# tell Docker what command we want to run when our image 
# is executed inside a container.
# CMD ["whatever", "you", "want"]