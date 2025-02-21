FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_UID=1000
ARG USER_GID=1000
ARG USERNAME=user

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update -y && apt-get install -y --allow-unauthenticated \
    git \
    vim \
    ranger \
    python3-pip \
    wget \
    time \
    && rm -rf /var/lib/apt/lists/*

# Set up Conda
ARG FORGE_VERSION=24.11.2-1
RUN wget --no-check-certificate -qO ~/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/download/${FORGE_VERSION}/Miniforge3-${FORGE_VERSION}-Linux-x86_64.sh && \
    chmod +x ~/miniforge.sh && \
    ~/miniforge.sh -b -p /opt/miniforge && \
    rm ~/miniforge.sh
ENV PATH=/opt/miniforge/bin:$PATH

# Clone repo
RUN mkdir workspace && cd workspace && git clone https://github.com/af-a/stable-imitation-policy --branch abcc-ds-extensions
RUN chown -R $USER_UID:$USER_GID workspace

# Set the default user to the newly created user
USER $USERNAME

# Set up Conda environment
ARG CONDA_ENV_NAME=learning-ds
RUN mamba env create -qy -f "workspace/stable-imitation-policy/environment.yaml"
RUN mamba init && echo "mamba activate ${CONDA_ENV_NAME}" >> /home/user/.bashrc

WORKDIR "/workspace/stable-imitation-policy/exp"
