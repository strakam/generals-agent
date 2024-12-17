# Use official Ubuntu as base image
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory for setup tasks
WORKDIR /workspace

# Install system dependencies including sudo
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    curl \
    git \
    ca-certificates \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with sudo privileges
RUN useradd -m -s /bin/bash -G sudo dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the new user
USER dev

# Set home directory environment variable
ENV HOME=/home/dev

# Install Python 3.12
RUN sudo -E wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz \
    && sudo tar -xzf Python-3.12.0.tgz \
    && cd Python-3.12.0 \
    && sudo ./configure --enable-optimizations \
    && sudo make -j$(nproc) \
    && sudo make altinstall \
    && cd .. \
    && sudo rm -rf Python-3.12.0.tgz Python-3.12.0

# Ensure pip is installed and updated
RUN sudo python3.12 -m ensurepip --upgrade

# Install Python packages
RUN sudo python3.12 -m pip install --no-cache-dir \
    torch \
    lightning \
    pettingzoo \
    gymnasium \
    wandb \
    numpy \
    matplotlib \
    neptune \
    scipy

# Set Python 3.12 as default
RUN sudo ln -sf /usr/local/bin/python3.12 /usr/bin/python3 \
    && sudo ln -sf /usr/local/bin/pip3.12 /usr/bin/pip3

# Set up bash profile and Zsh
# RUN echo "export PATH=/usr/local/bin:$PATH" >> ~/.bashrc \
#     && echo "cd $HOME" >> ~/.bashrc \
#     && sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
#     -t robbyrussell

# Set default shell to Zsh
RUN sudo chsh -s /bin/zsh dev

# Set the home directory as the default working directory
WORKDIR $HOME

# Set the default command to start Zsh
# CMD [ "zsh" ]
CMD ["/bin/bash"]

