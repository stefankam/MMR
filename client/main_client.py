#!/usr/bin/env python3
"""
main_client.py
Each client container runs this Flask app, registers with topology server, and exposes /train
endpoint.
"""
import os
import time
import base64
import io
import json
import random
import threading
from typing import Dict
import argparse
import requests
import time
import threading

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import copy

from topology_client import build_benign_prompts, poison_prompts, biased_data

app = Flask(__name__)

# defaults from environment
MODEL_NAME = os.environ.get("MODEL_NAME", "distilgpt2")
TOPOLOGY_URL = os.environ.get("TOPOLOGY_URL", "http://topology_server:8080")
CLIENT_PORT = int(os.environ.get("CLIENT_PORT", "5000"))
CLIENT_HOST = os.environ.get("CLIENT_HOST", "0.0.0.0")

# Local client state
client_id = None
address = None
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
# prepare local datasets (benign always available)
local_benign = build_benign_prompts(num_per_client=40)
# some clients may have malicious data available
has_malicious_data = random.random() < 1.0
if has_malicious_data:
    local_malicious = poison_prompts(local_benign, strength=0.6)
else:
    local_malicious = None

# Simple dataset wrapper
class TextPromptDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=64):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
    def __len__(self):
        return len(self.enc["input_ids"])
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k,v in self.enc.items()}

def state_dict_to_b64(state_dict):
    bio = io.BytesIO()
    torch.save(state_dict, bio)
    bio.seek(0)
    return base64.b64encode(bio.read()).decode("ascii")

def b64_to_state_dict(s):
    bio = io.BytesIO(base64.b64decode(s.encode("ascii")))
    bio.seek(0)
    return torch.load(bio, map_location="cpu")


def _read_proc_status() -> Dict[str, str]:
    """Read /proc/self/status into a key/value map."""
    out: Dict[str, str] = {}
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def get_resource_usage() -> Dict[str, Dict[str, str]]:
    """
    Resource telemetry derived from /proc/self/status.
    CPU is approximated via scheduler/context-switch indicators exposed in status.
    """
    status = _read_proc_status()
    return {
        "cpu": {
            "threads": status.get("Threads", "unknown"),
            "cpus_allowed_list": status.get("Cpus_allowed_list", "unknown"),
            "voluntary_ctxt_switches": status.get("voluntary_ctxt_switches", "unknown"),
            "nonvoluntary_ctxt_switches": status.get("nonvoluntary_ctxt_switches", "unknown"),
        },
        "memory": {
            "vm_size": status.get("VmSize", "unknown"),
            "vm_rss": status.get("VmRSS", "unknown"),
            "vm_hwm": status.get("VmHWM", "unknown"),
            "vm_swap": status.get("VmSwap", "unknown"),
        },
        "disk": {
            "rss_file": status.get("RssFile", "unknown"),
            "rss_shmem": status.get("RssShmem", "unknown"),
        },
    }



def apply_attack_to_delta(delta: Dict[str, torch.Tensor], attack_type: str, params: Dict[str, float]) -> Dict[str, torch.Tensor]:
    # Non-destructive clone
    out = {k: v.clone() for k, v in delta.items()}

    if attack_type == "scaled":
        s = float(params.get("scale_s", 3.0))
        for k in out.keys():
            out[k].mul_(s)
        return out

    if attack_type == "slow_drift":
        eps = float(params.get("slow_eps", 1e-3))
        # Add a tiny consistent bias to a subset (e.g., last 5%) of parameters
        for k in out.keys():
            v = out[k]
            n = v.numel()
            m = max(1, int(0.05 * n))
            v.view(-1)[-m:] += eps
        return out

    if attack_type == "backdoor":
        # Proxy: create sparse, directional offset on lm_head to emulate a trigger bias
        for k in out.keys():
            if "lm_head.weight" in k or "wte.weight" in k:
                v = out[k]
                n = v.numel()
                m = max(1, int(0.01 * n))
                v.view(-1)[:m] += 1e-2
        return out

    return out




@app.route("/train", methods=["POST"])
def train_endpoint():
    """
    Expects JSON:
    {
      "model_state_b64": "...",
      "use_malicious": bool,
      "local_epochs": int,
      "batch_size": int,
      "lr": float
    }
    Returns:
    { "delta_b64": "..." }
    """
    j = request.get_json()
    model_b64 = j["model_state_b64"]
    use_malicious = bool(j.get("use_malicious", False))
    attack_type = j.get("attack_type", "slow_drift")
    attack_param = j.get("attack_param", {})
    local_epochs = int(j.get("local_epochs", 1))
    batch_size = int(j.get("batch_size", 4))
    lr = float(j.get("lr", 5e-5))

    state = b64_to_state_dict(model_b64)
    # instantiate model with same architecture
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    # --- now you can safely load server weights ---
    model.load_state_dict(state)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # choose dataset
    if use_malicious and local_malicious is not None:
        texts = local_malicious
    else:
        texts = local_benign

    dataset = TextPromptDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(local_epochs):
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=(input_ids!=tokenizer.pad_token_id).long().to(device), labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # compute delta: new_state - old_state
    new_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    old_state = state
    delta = {k: (new_state[k] - old_state[k]).cpu() for k in old_state.keys()}
    # apply if requested
    if use_malicious and attack_type != "none":
       delta = apply_attack_to_delta(delta, attack_type, attack_param)
    # encode and return
    delta_b64 = state_dict_to_b64(delta)
    usage = get_resource_usage()
    return jsonify({"delta_b64": delta_b64, "resource_usage": usage})

@app.route("/resource_usage", methods=["GET"])
def resource_usage_endpoint():
    return jsonify(get_resource_usage())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", required=True)
    parser.add_argument("--server_ip", required=True)
    parser.add_argument("--server_port", type=int, default=5000)
    parser.add_argument("--port", type=int, default=CLIENT_PORT)
    args = parser.parse_args()

    CLIENT_PORT = args.port

    def register_with_server():
        while True:
            try:
                register_payload = {
                "device_id": args.device_id,
                "address": f"{CLIENT_HOST}:{CLIENT_PORT}",
                "malicious": has_malicious_data
                }
                res = requests.post(f"http://{args.server_ip}:{args.server_port}/register",
                                    json=register_payload)
                if res.status_code == 200:
                    print(f"[CLIENT {args.device_id}] Registered successfully.")
                    break
            except Exception as e:
                print(f"[CLIENT {args.device_id}] Server not ready yet ({e}), retrying...")
                time.sleep(2)

    threading.Thread(target=register_with_server, daemon=True).start()

    app.run(host=CLIENT_HOST, port=CLIENT_PORT)
