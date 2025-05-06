#!/bin/bash
set -e


mkdir -p ~/.ssh


# Check if the SSH key already exists, if not, generate it
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating SSH key..."
    ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -N ""
fi

# Define target containers and IPs
nodes=("172.20.0.2" "172.20.0.3" "172.20.0.4")

# Loop over nodes and copy the SSH key to them
for node in "${nodes[@]}"; do
    echo "Copying SSH key to $node..."
    sshpass -p "root" ssh-copy-id -o StrictHostKeyChecking=no root@$node
done

# Success message
echo "SSH key setup complete for all nodes!"
