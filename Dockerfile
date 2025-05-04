FROM ubuntu:22.04

# Install required tools
RUN apt-get update && apt-get install -y \
    build-essential \
    mpich \
    libmetis-dev \
    gcc g++ \
    libomp-dev \
    cmake \
    nano \
    openssh-server \
    iputils-ping \
    sshpass \
    && rm -rf /var/lib/apt/lists/*



# SSH setup
RUN mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Set environment to prevent interactive prompts
# ENV DEBIAN_FRONTEND=noninteractive

# Copy the SSH setup script
COPY setup_ssh.sh /setup_ssh.sh

# Make it executable
RUN chmod +x /setup_ssh.sh

# Set up workspace
WORKDIR /workspace

# Expose SSH port (for MPI communication)
EXPOSE 22

# Start SSH and keep the container alive
CMD /usr/sbin/sshd -D & /setup_ssh.sh && tail -f /dev/null
