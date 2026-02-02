#!/usr/bin/env python3
"""
topology_server.py
Lightweight client registry and sampling helper used by the coordinator.
Exposes a minimal HTTP endpoint (optional) or can be used as library.
"""
import random
from typing import List, Dict

# For simplicity this module provides a TopologyServer class that the coordinator imports.
# We also provide an optional Flask endpoint that client containers could call to register.
# For this docker deployment we'll run a simple Flask server as the topology service.

from flask import Flask, request, jsonify
app = Flask(__name__)

class TopologyServer:
    def __init__(self, base_url=None):
        # clients: dict id -> {"address": "host:port", "meta": {...}}
        self.clients = {}
        self.next_id = 0

    def register_client(self, address: str, meta: dict=None) -> int:
        cid = self.next_id
        self.next_id += 1
        self.clients[cid] = {"id": cid, "address": address, "meta": meta or {}}
        print(f"[TOPOLOGY] Registered client {cid} @ {address}")
        return cid

    def list_clients(self) -> List[Dict]:
        return list(self.clients.values())

    def sample_clients_for_models(self, num_clients: int, K: int, clients_per_model: int, rho: float):
        """
        Return mapping model_i -> list of client entries (dicts with id/address)
        A deterministic simple sampling over registered clients (ignoring num_clients param).
        """
        all_clients = list(self.clients.values())
        mapping = {}
        for i in range(K):
            if len(all_clients) == 0:
                mapping[i] = []
                continue
            base = set([c["id"] for c in random.sample(all_clients, min(len(all_clients), clients_per_model))])
            # implement a simple overlap: reuse some of previous
            if i > 0 and rho > 0:
                prev_pool = set()
                for j in range(max(0, i-3), i):
                    prev_pool |= set([c["id"] for c in mapping[j]])
                num_replace = int(len(base) * rho)
                if len(prev_pool) > 0 and num_replace > 0:
                    replace_from_prev = random.sample(list(prev_pool), min(num_replace, len(prev_pool)))
                    base_list = list(base)
                    to_replace = random.sample(base_list, min(len(replace_from_prev), len(base_list)))
                    for old,new in zip(to_replace, replace_from_prev):
                        base.remove(old)
                        base.add(new)
            mapping[i] = [ self.clients[cid] for cid in sorted(list(base)) ]
        return mapping

    def penalize_client(self, client_id: int, amount: float = 0.1):
        if client_id in self.clients:
            rep = self.clients[client_id]["meta"].get("reputation", 1.0)
            rep = max(0.0, rep - amount)
            self.clients[client_id]["meta"]["reputation"] = rep
            print(f"[TOPOLOGY] Penalized client {client_id}, rep -> {rep:.2f}")

# Minimal Flask endpoints for container registration (so clients can register themselves)
topo = TopologyServer()

@app.route("/register", methods=["POST"])
def register():
    j = request.get_json()
    addr = j.get("address")
    meta = j.get("meta", {})
    cid = topo.register_client(addr, meta)
    return jsonify({"client_id": cid})

@app.route("/list_clients", methods=["GET"])
def list_clients():
    return jsonify(topo.list_clients())

if __name__ == "__main__":
    # run topology server when invoked as main (used in docker-compose)
    app.run(host="0.0.0.0", port=6000)
