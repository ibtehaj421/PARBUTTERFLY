
---

# 🚀 Running OpenMP + MPI in Docker (Multi-Container Setup)

This outlines the essential steps to compile and run a hybrid OpenMP + MPI program across multiple Docker containers. SSH and Docker networking are assumed to be preconfigured.

---

## 📦 1. Start the Docker Containers

Make sure you're in the project root directory (with the `docker-compose.yml` file):

```bash
docker-compose up -d
```

* `up -d`: Starts all containers in detached mode (in the background).
* This spins up `node1`, `node2`, and `node3` with pre-defined static IPs.

---

## 🔁 2. Enter the Primary Container (node1)

```bash
docker exec -it node1 sh
```

* `exec -it`: Starts an interactive shell session in the container.
* `sh`: Launches the default shell.

---

## 📂 3. Navigate to the Workspace

Assuming volumes are mounted to `/workspace`:

```bash
cd /workspace
```

* This brings you to your source code directory shared with the host.

---

## 🧪 4. Compile the MPI + OpenMP Program

```bash
mpicc -fopenmp -o test your_program.c
mpicxx -fopenmp -O2 -std=c++17 -o par parbutter.cpp
```

* `mpicc`: MPI C compiler wrapper.
* `-fopenmp`: Enables OpenMP support.
* `-o test`: Output binary named `test`.

---

## 🔗 5. Test SSH Connectivity Between Nodes (Optional)

Verify other nodes are reachable:

```bash
ssh node2 "echo node2 is reachable"
ssh node3 "echo node3 is reachable"
```

* These confirm passwordless SSH is working as expected across nodes.

---

## 🚦 6. Run the Program Using MPI

```bash
mpirun -np 3 --host node1,node2,node3 ./test
mpirun --host node1:2,node2:2,node3:2 -np 3 ./par
```

* `-np 3`: Total 3 MPI processes.
* `--host`: Specifies the exact nodes to use.
* `./test`: Path to the compiled executable.

---

## 🔚 7. Exit the Container

```bash
exit
```

* Leaves the container’s interactive shell session.

---

## 🛑 8. Stop All Containers

```bash
docker stop $(docker ps -q)
```

* Stops all running containers.

---

## 🧹 (Optional) Remove All Containers

```bash
docker rm $(docker ps -aq)
```

* Cleans up all stopped containers.

---

## ✅ Notes

* This setup assumes passwordless SSH was established via a startup script like `setup_ssh.sh`.
* Docker handles hostname resolution (`node1`, `node2`, `node3`) automatically via the custom network.

---

