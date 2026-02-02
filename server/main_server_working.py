#!/usr/bin/env python3
"""
main_server.py
Coordinator / Orchestrator for DMM FL simulation.
Exposes a minimal admin endpoint and runs training rounds by calling client /train endpoints.
"""
import base64
import io
import json
import random
import time
from typing import Dict, List, Tuple

import requests
import torch
import statistics
import copy

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from topology_server import TopologyServer  # local helper (same directory)

# ---------------------------
# CONFIG (flexible experiment control)
# ---------------------------
import os

# Core FL setup
MODEL_NAME = os.environ.get("MODEL_NAME", "distilgpt2")
ROUNDS = int(os.environ.get("ROUNDS", "3"))

# MMR parameters
NUM_MODELS = int(os.environ.get("MMR_NUM_MODELS", "3"))
CLIENTS_PER_MODEL = int(os.environ.get("CLIENTS_PER_MODEL", "4"))
SAMPLING_OVERLAP_RHO = float(os.environ.get("SAMPLING_OVERLAP_RHO", "0.2"))

# Detection cadence and checkpointing
DETECTION_CADENCE = int(os.environ.get("DETECTION_CADENCE", "1"))
CHECKPOINT_INTERVAL = int(os.environ.get("CHECKPOINT_INTERVAL", "5"))

# Server communication
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5000"))

# Attack config (for simulation)
SIMULATE_ATTACK = os.environ.get("SIMULATE_ATTACK", "True").lower() in ("1", "true", "yes")
ATTACK_START_ROUND = int(os.environ.get("ATTACK_START_ROUND", "2"))
ATTACK_END_ROUND = int(os.environ.get("ATTACK_END_ROUND", "999"))
NUM_BYZANTINE_CLIENTS = int(os.environ.get("NUM_BYZANTINE_CLIENTS", "3"))

# ---------------------------
# MMR Ablation flags
# ---------------------------

# If False → identical client groups (no rotation)
MMR_ROTATION = os.environ.get("MMR_ROTATION", "True").lower() in ("1", "true", "yes")

# If False → detect but don’t quarantine/rollback
MMR_DIVERGENCE_MITIGATION = os.environ.get("MMR_DIVERGENCE_MITIGATION", "True").lower() in ("1", "true", "yes")

# Optional: use FLANDERS detector (hybrid)
USE_FLANDERS = os.environ.get("USE_FLANDERS", "False").lower() in ("1", "true", "yes")

# Tracking detection performance (TPR / FPR)
TRACK_METRICS = os.environ.get("TRACK_METRICS", "True").lower() in ("1", "true", "yes")



# Simple utility: encode / decode state_dict to base64 str
def state_dict_to_b64(state_dict: Dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    b = buffer.read()
    return base64.b64encode(b).decode("ascii")

def b64_to_state_dict(s: str) -> Dict[str, torch.Tensor]:
    b = base64.b64decode(s.encode("ascii"))
    buffer = io.BytesIO(b)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu")

# ---------------------------
# Server class
# ---------------------------
class DMMCoordinator:
    def __init__(self, model_name: str, topology_url: str):
        self.device_registry =  {}
        self.topology = TopologyServer(topology_url)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model_name = model_name
        # instantiate K server-side models (same init)
        self.models = [ AutoModelForCausalLM.from_pretrained(model_name).cpu() for _ in range(NUM_MODELS) ]
        # 🔧 resize embeddings for all models
        for model in self.models: 
           model.resize_token_embeddings(len(self.tokenizer))

        self.checkpoints = {}  # round -> consensus state
        self.safe_round = 0
        self.quarantined = set()
        # probe prompts (server-owned)
        self.probe_prompts = [
            "Write a short sentence about machine learning.",
            "How do you make tea?",
            "Translate the following sentence:",
            "The backdoor phrase is"
        ]
        while len(self.probe_prompts) < 32:
            self.probe_prompts.append(random.choice(self.probe_prompts))
        # simple baseline for probe variance
        self.probe_variance_baseline = None


    def start_registration_server(self):
        from flask import Flask, request, jsonify
        import threading

        app = Flask(__name__)
  
        @app.route("/register", methods=["POST"])
        def register_client():
           data = request.json
           device_id = data["device_id"]
           addr = request.remote_addr
           is_malicious = bool(data.get("malicious", False))
           host, port = data["address"].split(":")
           self.device_registry[device_id] = {'ip': addr, 'port': port, 'malicious': is_malicious, 'last_seen': time.time()}
           print(f"[SERVER] Registered client: {device_id} ({addr}:{port})")
           return jsonify({"status": "ok"})

        # Run Flask in background
        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000), daemon=True).start()



    def get_state(self, idx:int):
        return {k: v.detach().cpu().clone() for k, v in self.models[idx].state_dict().items()}

    def set_state(self, idx:int, state):
        self.models[idx].load_state_dict(state)

    def consensus_state(self):
        # average non-quarantined models
        active = [i for i in range(NUM_MODELS) if i not in self.quarantined]
        if len(active) == 0:
            active = list(range(NUM_MODELS))
        states = [self.get_state(i) for i in active]
        agg = copy.deepcopy(states[0])
        for s in states[1:]:
            for k in agg.keys():
                agg[k] = agg[k] + s[k]
        for k in agg.keys():
            agg[k] = agg[k] / len(states)
        return agg

    def compute_pairwise_distances(self) -> List[List[float]]:
        flats = []
        for i in range(NUM_MODELS):
            sd = self.get_state(i)
            parts = [v.reshape(-1) for v in sd.values()]
            flats.append(torch.cat(parts))
        D = [[0.0]*NUM_MODELS for _ in range(NUM_MODELS)]
        for i in range(NUM_MODELS):
            for j in range(i+1, NUM_MODELS):
                d = torch.norm(flats[i] - flats[j]).item()
                D[i][j] = d
                D[j][i] = d
        return D

    def compute_probe_losses(self):
        losses = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(NUM_MODELS):
            m = copy.deepcopy(self.models[i]).to(device).eval()
            total = 0.0
            n = 0
            with torch.no_grad():
                for p in self.probe_prompts:
                    enc = self.tokenizer(p, return_tensors="pt", truncation=True, padding=True).to(device)
                    out = m(**enc, labels=enc["input_ids"])
                    total += out.loss.item()
                    n += 1
            losses.append(total / max(1, n))
        return losses

    def detect_anomalies(self, D, probe_losses, alpha=3.0, beta=2.0):
        # flatten upper triangle
        tri = []
        for i in range(NUM_MODELS):
            for j in range(i+1, NUM_MODELS):
                tri.append(D[i][j])
        if len(tri) == 0:
            return []
        med = statistics.median(tri)
        mad = statistics.median([abs(x-med) for x in tri]) if len(tri)>0 else 0.0
        threshold = med + alpha * (mad if mad>0 else 1e-6)
        flags = []

        for i in range(NUM_MODELS):
            for j in range(i+1, NUM_MODELS):
                if D[i][j] > threshold:
                    flags.append(("pairwise-divergence", i,j, D[i][j], threshold))

        var_probe = statistics.pvariance(probe_losses) if len(probe_losses)>1 else 0.0
        if self.probe_variance_baseline is None:
            self.probe_variance_baseline = max(var_probe, 1e-6)
        if var_probe > beta * self.probe_variance_baseline:
            flags.append(("probe-variance-spike", i, var_probe, self.probe_variance_baseline))

        # simple cluster-split heuristic
        if max(tri) > 4.0 * med + 1e-6:
            mx = max(tri)
            for i in range(NUM_MODELS):
                for j in range(i+1, NUM_MODELS):
                    if D[i][j] == mx:
                        flags.append(("cluster-split", i,j,mx))
                        break
                else:
                    continue
                break
        return flags

    def respond_to_flags(self, flags, round_idx, client_contribs, client_updates):
        print(f"[SERVER] Round {round_idx}: flags = {flags}")
        safe_round = max(self.checkpoints.keys()) if len(self.checkpoints)>0 else None
        for f in flags:
            if f[0] in ("pairwise-divergence","cluster-split"):
                _, i, j, *_ = f
                offenders = {i,j}
                for m in offenders:
                    print(f"[SERVER] Quarantine & rollback model {m}")
                    self.quarantined.add(m)
                    if safe_round is not None:
                        state = self.checkpoints[safe_round]
                        self.set_state(m, state)
                        print(f"[SERVER] Rolled back model {m} to checkpoint {safe_round}")
                    # find top contributing clients by norm
                    contribs = client_contribs.get(m, [])
                    scored = []
                    for cid in contribs:
                        tup = (m, cid)
                        delta_b64 = client_updates.get(tup)
                        if delta_b64 is None: continue
                        delta_state = b64_to_state_dict(delta_b64)
                        parts = [v.reshape(-1) for v in delta_state.values()]
                        vec = torch.cat(parts)
                        scored.append((cid, float(torch.norm(vec))))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    top = [cid for cid,_ in scored[:min(5,len(scored))]]
                    for cid in top:
                        self.topology.penalize_client(cid, amount=0.2)

            elif f[0] == "probe-variance-spike":
                print("[SERVER] Probe variance spike — increasing monitoring")
                # update baseline to be more conservative
                self.probe_variance_baseline = (self.probe_variance_baseline + f[1]) / 2.0

    def aggregate_and_apply(self, updates_list: List[Dict[str,str]], method="mean"):
        """
        updates_list: list of base64-encoded state-dict deltas (strings)
        returns aggregated delta state
        """
        tensors_list = [ b64_to_state_dict(b64) for b64 in updates_list ]
        # simple mean aggregation
        agg = copy.deepcopy(tensors_list[0])
        for k in agg.keys():
            agg[k] = agg[k].clone()
        for t in tensors_list[1:]:
            for k in agg.keys():
                agg[k] = agg[k] + t[k]
        for k in agg.keys():
            agg[k] = agg[k] / len(tensors_list)
        return agg

    def run_rounds(self, rounds=ROUNDS):
        # initial checkpoint
        self.checkpoints[0] = self.consensus_state()
        self.safe_round = 0

        # Use directly registered clients (no topology)
        clients_info = [
        {
        "id": cid,
        "address": client_entry["ip"],
        "port": int(client_entry["port"])
        }
        for cid, client_entry in self.device_registry.items()
        ]

#        clients_info = self.topology.list_clients()
        print(f"[SERVER] Known clients: {clients_info}")

        for r in range(1, rounds+1):
            print(f"\n[SERVER] === Round {r} ===")
 #           mapping = self.topology.sample_clients_for_models(len(clients_info), NUM_MODELS, CLIENTS_PER_MODEL, SAMPLING_OVERLAP_RHO)
            # mapping: dict model_idx -> list of client_ids (the topology server returns addresses)
            model_updates = {i: [] for i in range(NUM_MODELS)}
            client_updates = {}   # (model,client) -> base64 delta
            client_contribs = {i: [] for i in range(NUM_MODELS)}

            # request updates from clients
            for i in range(NUM_MODELS):

                if MMR_ROTATION:
                   # normal rotation sampling (diverse subsets)
                   sampled = random.sample(clients_info, k=min(CLIENTS_PER_MODEL, len(clients_info)))
                else:
                   # reuse identical group (no rotation)
                   if r == 1:
                      global shared_sample_group
                      shared_sample_group = random.sample(clients_info, k=min(CLIENTS_PER_MODEL, len(clients_info)))
                   sampled = shared_sample_group


                for client_entry in sampled:
                    # client_entry is a dict with id and address, our topology_server will provide this
                    cid = client_entry["id"]
                    addr = client_entry["address"]
                    port = client_entry["port"]
                    # decide whether to request malicious training mode (server-side toggle)
                    use_malicious = False
                    cid_num = int(cid.split("_")[-1])
                    if SIMULATE_ATTACK and (ATTACK_START_ROUND <= r <= ATTACK_END_ROUND) and cid_num < NUM_BYZANTINE_CLIENTS:
                        use_malicious = True
                    # prepare model payload (send state)
                    state_b64 = state_dict_to_b64(self.get_state(i))
                    payload = {
                        "model_state_b64": state_b64,
                        "use_malicious": use_malicious,
                        "local_epochs": 1,
                        "batch_size": 4,
                        "lr": 5e-5
                    }
                    # POST to client /train endpoint
                    url = f"http://{addr}:{port}/train"
                    try:
                        resp = requests.post(url, json=payload, timeout=60)
                        resp.raise_for_status()
                        result = resp.json()
                        delta_b64 = result["delta_b64"]
                        model_updates[i].append(delta_b64)
                        client_updates[(i,cid)] = delta_b64
                        client_contribs[i].append(cid)
                        print(f"[SERVER] Received update from client {cid} for model {i}")
                    except Exception as e:
                        print(f"[SERVER] Error contacting client {cid} at {addr}: {e}")

            # per-model aggregation and apply
            for i in range(NUM_MODELS):
                if i in self.quarantined:
                    print(f"[SERVER] Model {i} is quarantined — skip aggregation")
                    continue
                if len(model_updates[i]) == 0:
                    continue
                agg_delta = self.aggregate_and_apply(model_updates[i], method="mean")
                # apply to server-side model i
                # new_state = old + delta  (delta crafted as new - old by client)
                old_state = self.get_state(i)
                new_state = {}
                for k in old_state.keys():
                    new_state[k] = old_state[k] + agg_delta[k]
                self.set_state(i, new_state)

            # checkpoint
            if r % CHECKPOINT_INTERVAL == 0:
                self.checkpoints[r] = self.consensus_state()
                self.safe_round = r
                print(f"[SERVER] Stored safe checkpoint at round {r}")

            # detection
            if r % DETECTION_CADENCE == 0:
                D = self.compute_pairwise_distances()
                probe_losses = self.compute_probe_losses()
                flags = self.detect_anomalies(D, probe_losses)
                
                # --- Initialize storage for this round ---
                if not hasattr(self, "round_flags"):
                   self.round_flags = {}
                self.round_flags[r] = []

                # --- Log which clients/models were flagged ---
                if flags:
                   print(f"[SERVER] Round {r}: Detected anomalies ({len(flags)} flags) → {flags}")
                   detected_clients = set()

                   for f in flags:
                       # Example: ('pairwise-divergence', 0, 1, ...)
                       if f[0] == "pairwise-divergence":
                          _, i, j, *_ = f
                          detected_clients.add(f"Device_{i}")
                          detected_clients.add(f"Device_{j}")
                       elif f[0] == "probe-variance-spike":
                          _, i, *_ = f
                          detected_clients.add(f"Device_{i}")
                       elif isinstance(f, tuple) and len(f) == 2 and isinstance(f[1], bool):
                          cid, flag = f
                          if flag:
                             detected_clients.add(cid)
 
                   self.round_flags[r] = list(detected_clients)
                   print("[DEBUG] Final round_flags =", self.round_flags)

                   if MMR_DIVERGENCE_MITIGATION:
                      self.respond_to_flags(flags, r, client_contribs, client_updates)
                   else:
                      print(f"[SERVER] Detected anomalies but mitigation disabled (flags={flags})")
                else:
                    # update baseline slowly
                    varp = statistics.pvariance(probe_losses) if len(probe_losses)>1 else 0.0
                    if self.probe_variance_baseline is None:
                        self.probe_variance_baseline = max(varp, 1e-6)
                    else:
                        self.probe_variance_baseline = 0.9*self.probe_variance_baseline + 0.1*varp
                    print("[SERVER] No anomalies detected.")

        if TRACK_METRICS:
           TP = FP = FN = TN = 0
           for cid, info in self.device_registry.items():
               gt_malicious = info.get("malicious", False)
               print("gt_malicious = info.get returns:", gt_malicious) 
               detected_any = any(
                 cid in self.round_flags.get(r, [])
                 for r in range(1, rounds+1) 
               )

           if gt_malicious and detected_any:
              TP += 1
           elif gt_malicious and not detected_any:
              FN += 1
           elif not gt_malicious and detected_any:
              FP += 1
           else:
              TN += 1
           TPR = TP / (TP + FN + 1e-6)
           FPR = FP / (FP + TN + 1e-6)
           print(f"[SERVER] Client-level Detection: TPR={TPR:.3f}, FPR={FPR:.3f}")

        print("[SERVER] Training rounds complete.")


if __name__ == "__main__":
    topology_url = "http://topology_server:8080"
    server = DMMCoordinator(MODEL_NAME, topology_url)

    # --- Start registration listener (Flask app runs in a background thread) ---
    server.start_registration_server()  # if your class doesn’t have this, add it (see below)

    # --- Wait for clients to register before starting rounds ---
    EXPECTED_CLIENTS = 3  # or however many you’re starting
    import time
    while len(server.device_registry) < EXPECTED_CLIENTS:
        print(f"[SERVER] Waiting for clients... ({len(server.device_registry)}/{EXPECTED_CLIENTS})")
        time.sleep(5)

    print(f"[SERVER] All clients registered: {list(server.device_registry.keys())}")

    # --- Then start training rounds ---
    server.run_rounds(rounds=ROUNDS)
