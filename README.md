# cuda-container-vscode

Using GPUs in VS Code with Nvidia CUDA Container

A docker container with GPUs 

## Make sure you have a GPU 

```bash
nvidia-smi

Wed Dec  7 23:03:30 2022    
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.141.03   Driver Version: 470.141.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:09:00.0 Off |                  N/A |
|  0%   44C    P8    29W / 350W |     19MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1448      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      1683      G   /usr/bin/gnome-shell                8MiB |
+-----------------------------------------------------------------------------+
```   


## Install Docker Image 

__The version of Image has to be same with your CUDA version__ and architecture
of your computer (amd64 for my case). 

Mine is `CUDA Version: 11.4`, therefore I need this image: `11.4.2-cudnn8-devel-ubuntu20.04`

Please follow the instruction here: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

To test whether you installed the docker image or not, run

```bash
docker run --rm --gpus all nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04 nvidia-smi

Thu Dec  8 08:20:52 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.141.03   Driver Version: 470.141.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:09:00.0 Off |                  N/A |
|  0%   43C    P8    29W / 350W |     19MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

In docker hub, Nvidia/cuda provides many images: https://hub.docker.com/r/nvidia/cuda

## Setup development container in VS Code

Please make sure you will put `requirements.txt` into the same directory
with your `Dockerfile`. Otherwise you will have change the reference path 
in your Dockerfile. 

## Remove Dangling aka <none> Images

`docker rmi $(docker images -f "dangling=true" -q)`
