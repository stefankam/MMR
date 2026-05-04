# Model Multiplicity Rotation (MMR)

This repository implements a **federated learning (FL) framework with Model Multiplicity Rotation (MMR)** for detecting adversarial behavior during training.

MMR maintains **multiple rotating global models** and detects anomalies through **cross-model divergence**, enabling robust detection of:

* backdoor attacks
* scaled-gradient
* slow-drift 

and more

---

## ⚙️ Overview

* **Multiple models (K)** trained in parallel
* **Client rotation across models**
* **Divergence-based anomaly detection**
* **Automatic mitigation (rollback / filtering)**

Communication is implemented via Flask and deployment uses Docker.

---

## 🚀 Quick Start

### 1. Build Containers

```bash
docker build -t mmr_server ./server
docker build -t mmr_client ./client
```

---

### 2. Run Server

```bash
docker run --rm -p 8080:8080 --name mmr_server mmr_server
```

---

### 3. Launch Anonymous Client Containers (Remote)

Example script to spawn multiple **anonymous FL clients** via SSH + tmux:

```bash
#!/bin/bash

SSH_USER=***
SSH_HOST=***

SERVER_IP="172.17.0.2"
BASE_PORT=5000
NUM_CLIENTS=10

for ((i=0; i<$NUM_CLIENTS; i++)); do
  DEVICE_ID="Device_$i"
  PORT=$((BASE_PORT))

osascript <<EOF
tell application "Terminal"
  do script "ssh ${SSH_USER}@${SSH_HOST} \"tmux kill-session -t MMR_job_$i 2>/dev/null; tmux new-session -d -s MMR_job_$i 'docker run --rm \
    mmr_client python3 main_client.py --device_id ${DEVICE_ID} --port ${PORT} --server_ip ${SERVER_IP} > /tmp/MMR_job_$i.log 2>&1'\""
end tell
EOF

  sleep 5
done
```

### What this does

* Spawns multiple remote clients
* Each runs in an isolated Docker container
* Logs stored in `/tmp/MMR_job_<id>.log`
* Clients register automatically with the server

---

## 🔁 Workflow

1. Server initializes multiple models
2. Clients are assigned and rotated across models
3. Local training is performed
4. Server aggregates per-model updates
5. Divergence across models is monitored
6. Anomalies are detected and mitigated

---

## 📊 Outputs

* Model divergence metrics
* Detection signals (AUC, TTD)
* Training logs per client

---

## 📜 Notes

* Designed for **research experiments**
* Supports **scaling via containerized clients**
* Detection sensitivity depends on configuration

---


