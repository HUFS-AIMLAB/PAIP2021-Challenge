FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ADD . .

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
RUN apt-get install -y wget \
    python3-pip \
    curl \
    htop \
    vim \
    git \ 
    tree

RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get install -y openslide-tools
RUN apt-get install -y python-openslide
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install openslide-python && \
    pip install jupyter notebook && \
    pip install matplotlib && \
    pip install seaborn && \
    pip install timm  && \
    pip install segmentation-models-pytorch && \
    pip install albumentations && \
    pip install monai && \
    pip install tqdm && \
    pip install pillow && \
    pip install opencv-python && \
    pip install scikit-image

EXPOSE 8888