# -----------------------------------------------------------------------------
# Base image
# -----------------------------------------------------------------------------
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
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
WORKDIR /HabitatLLM
VOLUME ["/HabitatLLM"]

COPY . /HabitatLLM

# Download submodules
RUN git submodule update --init --recursive \
    && cd third_party/habitat-lab \
    && git fetch --tags \
    && git checkout v0.3.3 \
    && cd ../.. \
    && cd third_party/partnr-planner \
    && git checkout 2eb0e71 \
    && cd ../..

# -----------------------------------------------------------------------------
# Create, activate conda environment and install dependencies
# -----------------------------------------------------------------------------

# Create conda environment
RUN conda create -n habitat python=3.9.2 cmake=3.14.0 -y
RUN conda init bash


# Install conda packages (설치 channel 때문에 건드리기가 좀 어려운듯)
RUN conda run -n habitat conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
RUN conda run -n habitat conda install numpy=1.26.4 huggingface_hub -y
RUN conda run -n habitat conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y
RUN conda run -n habitat conda install -c conda-forge ffmpeg=4.3.1 -y

# Install pip requirements
RUN conda run -n habitat pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Create users and give permissions
# -----------------------------------------------------------------------------
# bwoo 8002:1002
# intern 8008:1002
# lune 8016:1002
# dongjin 8020:1002
# kimsh 8024:1002
RUN useradd -m -u 8018 -s /bin/bash taeyoon && echo "taeyoon:0000" | chpasswd && usermod -aG sudo taeyoon \
    && useradd -m -u 8002 -s /bin/bash bwoo && echo "bwoo:0000" | chpasswd && usermod -aG sudo bwoo \
    && useradd -m -u 8008 -s /bin/bash intern && echo "intern:0000" | chpasswd && usermod -aG sudo intern \
    && useradd -m -u 8016 -s /bin/bash lune && echo "lune:0000" | chpasswd && usermod -aG sudo lune \
    && useradd -m -u 8020 -s /bin/bash dongjin && echo "dongjin:0000" | chpasswd && usermod -aG sudo dongjin \
    && useradd -m -u 8024 -s /bin/bash kimsh && echo "kimsh:0000" | chpasswd && usermod -aG sudo kimsh

# Set root password
RUN echo "root:0000" | chpasswd

# Give permissions to all users
RUN chown -R :sudo /opt/conda && chmod -R g+rwx /opt/conda && \
    chown -R :sudo /workspace && chmod -R g+rwx /workspace && \
    chown -R :sudo /HabitatLLM && chmod -R g+rwx /HabitatLLM

# Set default permissions for new files for all users
RUN echo "umask 0002" >> /etc/bash.bashrc && \
    echo "umask 0002" >> /root/.bashrc && \
    for user in taeyoon bwoo intern lune dongjin kimsh; do \
        echo "umask 0002" >> /home/$user/.bashrc; \
    done

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
WORKDIR /HabitatLLM

SHELL ["conda", "run", "-n", "habitat", "/bin/bash", "-c"]