# -----------------------------------------------------------------------------
# Base image
# -----------------------------------------------------------------------------
FROM --platform=linux/amd64 nvidia/cuda:12.4.0-runtime-ubuntu22.04
# workspace and data directory
RUN mkdir -p /workspace /data

# -----------------------------------------------------------------------------
# Install system prerequisites
# -----------------------------------------------------------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl git wget g++ sudo vim \
    libegl1 libgles2 libopengl0 libgl1 libxrandr2 libxinerama1 libxcursor1 libosmesa6 libglvnd0 libglx0 libx11-6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Mount the data directory
VOLUME ["/data"]
ENV HF_HUB_CACHE /data/huggingface_cache
ENV HABITAT_DATASET_PATH /data

# -----------------------------------------------------------------------------
# Install conda
# -----------------------------------------------------------------------------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda -u && \
    rm ~/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# -----------------------------------------------------------------------------
# Copy repository codes and mount volumes
# -----------------------------------------------------------------------------
WORKDIR /MEMENTO
VOLUME ["/MEMENTO"]

COPY . /MEMENTO

# Download submodules
RUN git submodule sync \
    && git submodule update --init --recursive \
    && cd third_party/habitat-lab \
    && git fetch --tags \
    && git checkout v0.3.3 \
    && cd ../.. \
    && cd third_party/partnr-planner \
    && git checkout main \
    && cd ../..

# -----------------------------------------------------------------------------
# Create, activate conda environment and install dependencies
# -----------------------------------------------------------------------------

# Create conda environment
RUN conda create -n habitat python=3.9.2 cmake=3.14.0 -y
RUN conda init bash


# Install conda packages
RUN conda run -n habitat conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
RUN conda run -n habitat conda install numpy=1.26.4 huggingface_hub -y
RUN conda run -n habitat conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y
RUN conda run -n habitat conda install -c conda-forge ffmpeg=4.3.1 -y



# -----------------------------------------------------------------------------
# Install pip requirements
# -----------------------------------------------------------------------------
RUN conda run -n habitat pip install -r requirements.txt

# -----------------------------------------------------------------------------
# OpenGL
# -----------------------------------------------------------------------------
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{ \
        "file_format_version" : "1.0.0", \
        "ICD" : { \
            "library_path" : "libEGL_nvidia.so.0" \
        } \
    }' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# -----------------------------------------------------------------------------
# Final setup
# -----------------------------------------------------------------------------
WORKDIR /MEMENTO

SHELL ["conda", "run", "-n", "habitat", "/bin/bash", "-c"]