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
import csv
import math
import time
from typing import Dict, List, Tuple
import os
import requests
import torch
import statistics
import copy
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from topology_server import TopologyServer  # local helper (same directory)
from flanders_strategies.aggregate import aggregate_dnc
from dataclasses import dataclass
# ---------------------------
# CONFIG (flexible experiment control)
# ---------------------------

# Core FL setup
MODEL_NAME = os.environ.get("MODEL_NAME", "distilgpt2")
ROUNDS = int(os.environ.get("ROUNDS", "10"))

# MMR parameters
NUM_MODELS = int(os.environ.get("MMR_NUM_MODELS", "5"))
CLIENTS_PER_MODEL = int(os.environ.get("CLIENTS_PER_MODEL", "4"))
SAMPLING_OVERLAP_RHO = float(os.environ.get("SAMPLING_OVERLAP_RHO", "0.2"))

# Detection cadence and checkpointing
DETECTION_CADENCE = int(os.environ.get("DETECTION_CADENCE", "1"))
CHECKPOINT_INTERVAL = int(os.environ.get("CHECKPOINT_INTERVAL", "5"))

# Server communication
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5000"))
CLIENT_REQUEST_TIMEOUT = int(os.environ.get("CLIENT_REQUEST_TIMEOUT", "300"))

# Trim ratio for robust aggregation (drop ratio on each tail)
ROBUST_TRIM_RATIO = float(os.environ.get("ROBUST_TRIM_RATIO", "0.2"))


# Attack config (for simulation)
SIMULATE_ATTACK = os.environ.get("SIMULATE_ATTACK", "True").lower() in ("1", "true", "yes")
ATTACK_START_ROUND = int(os.environ.get("ATTACK_START_ROUND", "2"))
ATTACK_END_ROUND = int(os.environ.get("ATTACK_END_ROUND", "999"))
NUM_BYZANTINE_CLIENTS = int(os.environ.get("NUM_BYZANTINE_CLIENTS", "3"))


# Detector selection per run: {"NONE","MMR","FLANDERS"}
# (We keep your MMR pairwise+probe detector; FLANDERS stub below.)
ALPHA = float(os.environ.get("ALPHA", "3.0"))  # MMR MAD multiplier
BETA  = float(os.environ.get("BETA",  "2.0"))  # probe variance spike factor

# FLANDERS window length (sliding MAR residuals proxy)
FL_W_MIN = int(os.environ.get("FL_W_MIN", "5"))
FL_W_MAX = int(os.environ.get("FL_W_MAX", "10"))



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






# ---------------------------
# Experiment config objects
# ---------------------------
@dataclass
class ExperimentConfig:
    detector: str       # "NONE" | "MMR" | "FLANDERS" | "ROBUST"
    K_clients: int      # 100
    rounds: int         # 50
    q_participation: float   # 0.2/0.5/0.8
    p_attack: float     # 0.05/0.10/0.20
    Nm: int            # 1/2/3/5
    seed: int
    attack_type: str    # "slow_drift" | "scaled" | "backdoor"
    scale_s: float = 3.0         # for scaled-gradient
    slow_eps: float = 1e-3       # for slow-drift magnitude
    flanders_W: int = 8          # sliding window (5..10)
    rotation: bool = True        # MMR rotation (ρ used)
    rho_overlap: float = SAMPLING_OVERLAP_RHO
    mitigation: bool = True      # quarantine/rollback
    dirichlet_alpha: float = 1.0 # (placeholder if you later partition data)

# Per-round metrics recorder
class MetricsRecorder:
    def __init__(self, exp: ExperimentConfig, csv_path: str):
        self.exp = exp
        self.csv_path = csv_path
        self.rows = []   # one row per round
        self.round_flags: Dict[int, list] = {}
        self.round_scores: Dict[int, float] = {}  # detector score per round
        self.attack_active: Dict[int, bool] = {}  # ground truth (any attacker selected this round)
        self.first_detect_round = None            # for TTD
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        # Create file once and keep appending rows round-by-round for all experiment sweeps.
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "seed","detector","Nm","q","p","attack","round","y_true","score","flags",
                    "fpr",
                    "cpu_threads","cpu_allowed_list","cpu_voluntary_ctxt_switches","cpu_nonvoluntary_ctxt_switches",
                    "mem_vm_size","mem_vm_rss","mem_vm_hwm","mem_vm_swap",
                    "disk_rss_file","disk_rss_shmem",
                    "overhead_round_total_sec","overhead_detection_sec",
                    "run_tag","summary_auc","summary_ttd","record_type",
                ])


    def _append_round_row(
        self,
        r: int,
        resource_usage: Dict[str, Dict[str, str]],
        overhead: Dict[str, float],
        fpr: float,
        summary: Dict[str, any],
    ):
        y = 1 if self.attack_active[r] else 0
        s = self.round_scores[r]
        fl = self.round_flags[r]
        cpu = resource_usage.get("cpu", {})
        mem = resource_usage.get("memory", {})
        disk = resource_usage.get("disk", {})
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                self.exp.seed,
                self.exp.detector,
                self.exp.Nm,
                self.exp.q_participation,
                self.exp.p_attack,
                self.exp.attack_type,
                r,
                y,
                s,
                repr(fl),
                fpr,
                cpu.get("threads", "unknown"),
                cpu.get("cpus_allowed_list", "unknown"),
                cpu.get("voluntary_ctxt_switches", "unknown"),
                cpu.get("nonvoluntary_ctxt_switches", "unknown"),
                mem.get("vm_size", "unknown"),
                mem.get("vm_rss", "unknown"),
                mem.get("vm_hwm", "unknown"),
                mem.get("vm_swap", "unknown"),
                disk.get("rss_file", "unknown"),
                disk.get("rss_shmem", "unknown"),
                overhead.get("round_total_sec", 0.0),
                overhead.get("detection_sec", 0.0),
                self.run_tag(),
                summary.get("AUC"),
                summary.get("TTD"),
                "round",
            ])



    def run_tag(self) -> str:
        return (
            f"{self.exp.detector}_Nm{self.exp.Nm}_q{self.exp.q_participation}"
            f"_p{self.exp.p_attack}_{self.exp.attack_type}_seed{self.exp.seed}"
        )

    def append_summary_row(self, summary: Dict[str, any]):
        auc = summary.get("AUC")
        auc_cell = "" if auc is None else auc
        ttd = summary.get("TTD")
        ttd_cell = "" if ttd is None else ttd
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                self.exp.seed,
                self.exp.detector,
                self.exp.Nm,
                self.exp.q_participation,
                self.exp.p_attack,
                self.exp.attack_type,
                "",
                "",
                "",
                "",
                "","","","",
                "","","","",
                "","",
                "",
                "",
                self._run_tag,
                auc_cell,
                ttd_cell,
                "summary",
            ])


    def log_round(
        self,
        r: int,
        score: float,
        flags: List[tuple],
        attack_any: bool,
        resource_usage: Dict[str, Dict[str, str]],
        overhead: Dict[str, float],
    ):
        self.round_scores[r] = score
        self.round_flags[r] = flags
        self.attack_active[r] = attack_any
        if attack_any and flags and self.first_detect_round is None:
            self.first_detect_round = r
        y_all = [1 if self.attack_active[k] else 0 for k in sorted(self.attack_active.keys())]
        pred_all = [1 if self.round_flags[k] else 0 for k in sorted(self.round_flags.keys())]
        fp = sum(1 for yy, pp in zip(y_all, pred_all) if yy == 0 and pp == 1)
        tn = sum(1 for yy, pp in zip(y_all, pred_all) if yy == 0 and pp == 0)
        fpr = fp / (fp + tn + 1e-9)
        current_summary = self.summary()
        self._append_round_row(r, resource_usage, overhead, fpr, current_summary)


    def export_csv(self, fname: str):
        # ROC/AUC wants (y_true,y_score) per round; we also dump flags
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "seed","detector","Nm","q","p","attack","round","y_true","score","flags",
                "fpr",
                "cpu_threads","cpu_allowed_list","cpu_voluntary_ctxt_switches","cpu_nonvoluntary_ctxt_switches",
                "mem_vm_size","mem_vm_rss","mem_vm_hwm","mem_vm_swap",
                "disk_rss_file","disk_rss_shmem",
                "overhead_round_total_sec","overhead_detection_sec",
                "run_tag","summary_auc","summary_ttd","record_type",
            ])
            for r in sorted(self.round_scores.keys()):
                y = 1 if self.attack_active[r] else 0
                s = self.round_scores[r]
                fl = self.round_flags[r]
                y_all = [1 if self.attack_active[k] else 0 for k in sorted(self.attack_active.keys()) if k <= r]
                pred_all = [1 if self.round_flags[k] else 0 for k in sorted(self.round_flags.keys()) if k <= r]
                fp = sum(1 for yy, pp in zip(y_all, pred_all) if yy == 0 and pp == 1)
                tn = sum(1 for yy, pp in zip(y_all, pred_all) if yy == 0 and pp == 0)
                fpr = fp / (fp + tn + 1e-9)
                summ = self.summary()
                row = [
                    self.exp.seed, self.exp.detector, self.exp.Nm, self.exp.q_participation,
                    self.exp.p_attack, self.exp.attack_type, r, y, s, repr(fl),
                    fpr,
                    "unknown", "unknown", "unknown", "unknown",
                    "unknown", "unknown", "unknown", "unknown",
                    "unknown", "unknown", 0.0, 0.0,
                    self._run_tag, summ.get("AUC"), summ.get("TTD"), "round",
                ]
                w.writerow(row)

    def summary(self) -> Dict[str, any]:
        # Rank-based ROC-AUC with tie handling.
        y = [1 if self.attack_active[r] else 0 for r in sorted(self.round_scores.keys())]
        s = [self.round_scores[r] for r in sorted(self.round_scores.keys())]
        # compute ROC points
        thresholds = sorted(set(s), reverse=True)
        if not thresholds:
            return {"AUC": 0.0, "TTD": None}
        pts = []
        P = sum(y); N = len(y) - P
        for th in thresholds:
            tp = sum(1 for yy, ss in zip(y, s) if ss >= th and yy==1)
            fp = sum(1 for yy, ss in zip(y, s) if ss >= th and yy==0)
            tpr = tp / (P + 1e-9)
            fpr = fp / (N + 1e-9)
            pts.append((fpr, tpr))
        # sort by FPR and trapezoid
        pts = sorted(pts)
        auc = 0.0
        for (x1,y1),(x2,y2) in zip(pts, pts[1:]):
            auc += (x2 - x1) * (y1 + y2) / 2.0
        # Per-round episode TTD:
        # - starts at 0 on attack start round,
        # - increases while attack episode has no flag,
        # - returns to 0 once a flag appears for that episode.
        TTD = 0.0
        rounds_sorted = sorted(self.round_scores.keys())
        if rounds_sorted:
            attack_starts = []
            prev_attack = False
            for rr in rounds_sorted:
                cur_attack = bool(self.attack_active.get(rr, False))
                if cur_attack and not prev_attack:
                    attack_starts.append(rr)
                prev_attack = cur_attack

            if attack_starts:
                current_r = rounds_sorted[-1]
                latest_start = max(attack_starts)
                detected = any(
                    bool(self.round_flags.get(rr))
                    for rr in rounds_sorted
                    if latest_start <= rr <= current_r
                )
                if not detected:
                    TTD = float(max(0, current_r - latest_start))
        return {"AUC": auc, "TTD": TTD}



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
    CPU is represented by scheduler/context-switch indicators in status.
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


# ---------------------------
# Server class
# ---------------------------
class DMMCoordinator:
    def __init__(self, model_name: str, topology_url: str, exp: ExperimentConfig = None):
        self.device_registry =  {}
        self.topology = TopologyServer(topology_url)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.exp = exp
        # If no experiment passed, fallback to NUM_MODELS
        Nm = NUM_MODELS if self.exp is None else getattr(self.exp, "Nm", NUM_MODELS)

        # instantiate K server-side models (same init)
        self.models = [AutoModelForCausalLM.from_pretrained(model_name).cpu() for _ in range(Nm)]

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
           is_malicious = bool(data.get("malicious", False))
           host, port = data["address"].split(":")
           remote_addr = request.remote_addr
           # 0.0.0.0/127.0.0.1 are bind addresses, not routable callback targets from peer containers.
           # If the advertised host is non-routable, fallback to the observed remote address.
           if host in ("0.0.0.0", "127.0.0.1", "localhost"):
               host = remote_addr
           self.device_registry[device_id] = {'ip': host, 'port': port, 'malicious': is_malicious, 'last_seen': time.time()}
           print(f"[SERVER] Registered client: {device_id} ({host}:{port})")
           return jsonify({"status": "ok"})

        @app.route("/resource_usage", methods=["GET"])
        def resource_usage_endpoint():
            return jsonify(get_resource_usage())


        # Run Flask in background
        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000), daemon=True).start()



    def get_state(self, idx:int):
        return {k: v.detach().cpu().clone() for k, v in self.models[idx].state_dict().items()}

    def set_state(self, idx:int, state):
        self.models[idx].load_state_dict(state)

    def consensus_state(self):
        # average non-quarantined models
        active = [i for i in range(Nm) if i not in self.quarantined]
        if len(active) == 0:
            active = list(range(Nm))
        states = [self.get_state(i) for i in active]
        agg = copy.deepcopy(states[0])
        for s in states[1:]:
            for k in agg.keys():
                agg[k] = agg[k] + s[k]
        for k in agg.keys():
            agg[k] = agg[k] / len(states)
        return agg

    def compute_pairwise_distances(self) -> List[List[float]]:
        Nm = len(self.models)
        flats = []
        for i in range(Nm):
            sd = self.get_state(i)
            parts = [v.reshape(-1) for v in sd.values()]
            flats.append(torch.cat(parts))
        D = [[0.0]*Nm for _ in range(Nm)]
        for i in range(Nm):
            for j in range(i+1, Nm):
                d = torch.norm(flats[i] - flats[j]).item()
                D[i][j] = d
                D[j][i] = d
        return D

    def compute_probe_losses(self):
        losses = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(Nm):
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
        for i in range(Nm):
            for j in range(i+1, Nm):
                tri.append(D[i][j])
        if len(tri) == 0:
            return []
        med = statistics.median(tri)
        mad = statistics.median([abs(x-med) for x in tri]) if len(tri)>0 else 0.0
        threshold = med + alpha * (mad if mad>0 else 1e-6)
        flags = []

        for i in range(Nm):
            for j in range(i+1, Nm):
                if D[i][j] > threshold:
                    flags.append(("pairwise-divergence", i,j, D[i][j], threshold))

        var_probe = statistics.pvariance(probe_losses) if len(probe_losses)>1 else 0.0
        if self.probe_variance_baseline is None:
            self.probe_variance_baseline = max(var_probe, 1e-6)
        if var_probe > beta * self.probe_variance_baseline:
            # (flag_name, variance, baseline)
            flags.append(("probe-variance-spike", var_probe, self.probe_variance_baseline))

        # simple cluster-split heuristic
        if max(tri) > 4.0 * med + 1e-6:
            mx = max(tri)
            for i in range(Nm):
                for j in range(i+1, Nm):
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
                _, var_probe, _ = f
                self.probe_variance_baseline = (self.probe_variance_baseline + var_probe) / 2.0

    def aggregate_and_apply(self, updates_list: List[Dict[str,str]], method="mean"):
        """
        updates_list: list of base64-encoded state-dict deltas (strings)
        returns aggregated delta state
        """
        tensors_list = [ b64_to_state_dict(b64) for b64 in updates_list ]
        # simple mean aggregation
        agg = copy.deepcopy(tensors_list[0])


        if method == "trimmed_mean":
            # Coordinate-wise trimmed mean over client deltas.
            # If too few clients to trim, fallback to mean.
            n = len(tensors_list)
            trim_k = int(n * ROBUST_TRIM_RATIO)
            can_trim = (n - 2 * trim_k) > 0 and trim_k > 0
            for k in agg.keys():
                stacked = torch.stack([t[k] for t in tensors_list], dim=0)  # [n, ...]
                if can_trim:
                    sorted_vals, _ = torch.sort(stacked, dim=0)
                    trimmed = sorted_vals[trim_k:n - trim_k]
                    agg[k] = torch.mean(trimmed, dim=0)
                else:
                    agg[k] = torch.mean(stacked, dim=0)
            return agg

        # default: simple mean aggregation

        for k in agg.keys():
            agg[k] = agg[k].clone()
        for t in tensors_list[1:]:
            for k in agg.keys():
                agg[k] = agg[k] + t[k]
        for k in agg.keys():
            agg[k] = agg[k] / len(tensors_list)
        return agg


    def flanders_score(
        self,
        recent_client_deltas: List[Dict[str, torch.Tensor]],
        W: int,
        expected_malicious: int = 1,
    ) -> float:
        """
        FLANDERS/DnC-inspired score:
        1) Convert recent deltas to NDArrays format
        2) Run DnC aggregation (aggregate_dnc) to get robust center
        3) Score as max L2 distance from each update to robust center
        """
        window = recent_client_deltas[-W:]
        if len(window) < 2:
            return 0.0

        keys = list(window[0].keys())
        results = []
        flat_updates = []
        for sd in window:
            nds = [sd[k].detach().cpu().numpy() for k in keys]
            results.append((nds, 1))
            flat_updates.append(np.concatenate([a.reshape(-1) for a in nds], axis=0))

        num_mal = max(1, min(expected_malicious, len(results) - 1))
        robust_nds = aggregate_dnc(results, c=1.0, b=0, niters=1, num_malicious=num_mal)
        robust_center = np.concatenate([a.reshape(-1) for a in robust_nds], axis=0)

        dists = [float(np.linalg.norm(v - robust_center)) for v in flat_updates]
        return max(dists) if dists else 0.0



    def init_models(self):
        """(Re)initialize models according to self.exp.Nm"""
        Nm = getattr(self.exp, "Nm", NUM_MODELS) 
        self.models = [
            AutoModelForCausalLM.from_pretrained(self.model_name).cpu()
            for _ in range(self.exp.Nm)
        ]
        for m in self.models:
            m.resize_token_embeddings(len(self.tokenizer))
        print(f"[SERVER] Initialized {self.exp.Nm} models for detector={self.exp.detector}")


    def run_rounds(self, rounds=ROUNDS, exp: ExperimentConfig=None, csv_path: str = "signaltest_all_runs.csv"):
       # initial checkpoint
       self.checkpoints[0] = self.consensus_state()
       self.safe_round = 0

       # ✅ ADD THIS HERE
       mr = MetricsRecorder(exp, csv_path)

       # build all registered clients
       all_clients = [
            {"id": cid, "address": data["ip"], "port": int(data.get("port", 5000))}
            for cid, data in self.device_registry.items()
        ]
       if len(all_clients) == 0:
            print("[SERVER] No clients registered.")
            return
        
       # sample participation q
       m = max(1, int(exp.q_participation * len(all_clients)))
       active_clients = random.sample(all_clients, k=m)
       
       Nm = getattr(self.exp, "Nm", NUM_MODELS)
       recent_deltas_for_flanders = []

       # per-model sampling
       if exp.rotation:
            # per model, draw CLIENTS_PER_MODEL_DEFAULT from active_clients
            mapping = {
                i: random.sample(active_clients, k=min(CLIENTS_PER_MODEL, len(active_clients)))
                for i in range(Nm)
            }
       else:
            same = random.sample(active_clients, k=min(CLIENTS_PER_MODEL, len(active_clients)))
            mapping = {i: same for i in range(Nm)}

       recent_deltas_for_flanders = []

       for r in range(1, rounds+1):
        round_start = time.perf_counter()
        print(f"\n[SERVER] === Round {r} ===")  
        round_resource_usage = get_resource_usage()
        print(f"[SERVER] Resource usage: {round_resource_usage}")
        model_updates = {i: [] for i in range(Nm)}
        client_updates = {}
        client_contribs = {i: [] for i in range(Nm)}
        
        # any attacker this round?
        attack_any = False
        

        # request updates
        for i in range(Nm):
            sampled = mapping[i]
            for ce in sampled:
                cid, addr, port = ce["id"], ce["address"], ce["port"]
                # decide attacker per request (Bernoulli with p)
                use_mal = (random.random() < exp.p_attack)
                if use_mal: attack_any = True

                state_b64 = state_dict_to_b64(self.get_state(i))
                payload = {
                    "model_state_b64": state_b64,
                    "use_malicious": use_mal,
                    "attack_type": exp.attack_type,  # "slow_drift" | "scaled" | "backdoor"
                    "attack_param": {
                        "scale_s": exp.scale_s,
                        "slow_eps": exp.slow_eps
                    },
                    "local_epochs": 1,
                    "batch_size": 4,
                    "lr": 5e-5
                }
                url = f"http://{addr}:{port}/train"
                try:
                    resp = requests.post(url, json=payload, timeout=CLIENT_REQUEST_TIMEOUT)
                    resp.raise_for_status()
                    result = resp.json()
                    delta_b64 = result["delta_b64"]
                    model_updates[i].append(delta_b64)
                    client_updates[(i, cid)] = delta_b64
                    client_contribs[i].append(cid)
                    # keep raw delta for FLANDERS score if needed
                    if exp.detector == "FLANDERS":
                        recent_deltas_for_flanders.append(b64_to_state_dict(delta_b64))
                except Exception as e:
                    print(f"[SERVER] Error contacting client {cid} at {addr}:{port}: {e}")

            # per-model aggregation and apply
            for i in range(Nm):
                if i in self.quarantined:
                    print(f"[SERVER] Model {i} is quarantined — skip aggregation")
                    continue
                if len(model_updates[i]) == 0:
                    continue
                agg_method = "trimmed_mean" if exp.detector == "ROBUST" else "mean"
                agg_delta = self.aggregate_and_apply(model_updates[i], method=agg_method)

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
            
            # --- Detector selection & score ---
            detection_start = time.perf_counter()
            flags = []
            score = 0.0
            
            if exp.detector == "NONE":
                # no detection: score is 0; no flags
                pass
            
            elif exp.detector == "ROBUST":
                # Robust-agg baseline: do trimmed mean for aggregation (already aggregated above);
                # detection power expected low; still compute a soft score as 0
                pass
            
            elif exp.detector == "FLANDERS":
                # score = sliding-window variance over recent deltas
                W = max(FL_W_MIN, min(FL_W_MAX, exp.flanders_W))
                exp_mal = max(1, int(round(exp.p_attack * max(1, len(recent_deltas_for_flanders)))))
                score = self.flanders_score(recent_deltas_for_flanders, W, expected_malicious=exp_mal)

                # turn score into flags via simple threshold learned from EMA baseline
                if not hasattr(self, "fl_base"):
                    self.fl_base = max(score, 1e-9)
                if score > 3.0 * self.fl_base:
                    flags.append(("flanders-spike", score, self.fl_base))
                    if exp.mitigation:
                        # optional: quarantine most-offending model by last pairwise distance
                        pass
                # slowly update baseline
                self.fl_base = 0.9 * self.fl_base + 0.1 * score
            
            elif exp.detector == "MMR":
                D = self.compute_pairwise_distances()
                probe_losses = self.compute_probe_losses()
                # take a scalar score = max pairwise distance OR probe variance, whichever higher (signal)
                tri = []
                for a in range(Nm):
                    for b in range(a+1, Nm):
                        tri.append(D[a][b])
                max_pair = max(tri) if tri else 0.0
                varp = statistics.pvariance(probe_losses) if len(probe_losses) > 1 else 0.0
                score = float(max(max_pair, varp))
            
                flags = self.detect_anomalies(D, probe_losses, alpha=ALPHA, beta=BETA)
                if flags and exp.mitigation:
                    self.respond_to_flags(flags, r, client_contribs, client_updates)
                elif not flags:
                    # update MMR baseline slowly (your original)
                    varp_now = statistics.pvariance(probe_losses) if len(probe_losses)>1 else 0.0
                    if self.probe_variance_baseline is None:
                        self.probe_variance_baseline = max(varp_now, 1e-6)
                    else:
                        self.probe_variance_baseline = 0.9*self.probe_variance_baseline + 0.1*varp_now
            

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
                          # Global signal; no specific model/client offender is encoded.
                          pass
                       elif isinstance(f, tuple) and len(f) == 2 and isinstance(f[1], bool):
                          cid, flag = f
                          if flag:
                             detected_clients.add(cid)
 
                   mr.round_flags[r] = list(detected_clients)
                   print("[DEBUG] Final round_flags =", mr.round_flags)

                   if MMR_DIVERGENCE_MITIGATION:
                      self.respond_to_flags(flags, r, client_contribs, client_updates)
                   else:
                      print(f"[SERVER] Detected anomalies but mitigation disabled (flags={flags})")

                   TP = FP = FN = TN = 0
                   for cid, info in self.device_registry.items():
                       gt_malicious = info.get("malicious", False)
                       print("gt_malicious = info.get returns:", gt_malicious) 
                       detected_any = any(
                         cid in mr.round_flags.get(r, [])
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

                else:
                    # update baseline slowly
                    varp = statistics.pvariance(probe_losses) if len(probe_losses)>1 else 0.0
                    if self.probe_variance_baseline is None:
                        self.probe_variance_baseline = max(varp, 1e-6)
                    else:
                        self.probe_variance_baseline = 0.9*self.probe_variance_baseline + 0.1*varp
                    print("[SERVER] No anomalies detected.")




            # log metrics
            detection_elapsed = time.perf_counter() - detection_start
            round_total_elapsed = time.perf_counter() - round_start
            mr.log_round(
                r,
                score,
                flags,
                attack_any,
                round_resource_usage,
                {"round_total_sec": round_total_elapsed, "detection_sec": detection_elapsed},
            )


        print("[SERVER] Training rounds complete.")
        # Metrics are already appended per-round to the combined csv_path.
        tag = f"{exp.detector}_Nm{exp.Nm}_q{exp.q_participation}_p{exp.p_attack}_{exp.attack_type}_seed{exp.seed}"
        summ = mr.summary()
        auc_str = "NA" if summ["AUC"] is None else f"{summ['AUC']:.3f}"
        print(f"[SERVER] {tag}  AUC={auc_str}  TTD={summ['TTD']}  csv={csv_path}")




if __name__ == "__main__":
    topology_url = "http://topology_server:8080"
    server = DMMCoordinator(MODEL_NAME, topology_url)   # no exp yet

    # --- Start registration listener once ---
    server.start_registration_server()

    EXPECTED_CLIENTS = 3
    while len(server.device_registry) < EXPECTED_CLIENTS:
        print(f"[SERVER] Waiting for clients... ({len(server.device_registry)}/{EXPECTED_CLIENTS})")
        time.sleep(5)
    print(f"[SERVER] All clients registered: {list(server.device_registry.keys())}")

    qs = [0.2]
    ps = [0.2]
    nms = [3]
    atks = ["backdoor"]
    seeds = [1]

    # Choose detector per sweep (repeat this block for each method: NONE, ROBUST, FLANDERS, MMR)
    # Loop over detectors and experiment configs
    all_results_csv = "signaltest_all_runs.csv"
    if os.path.exists(all_results_csv):
        os.remove(all_results_csv)


    for DET in ["MMR", "ROBUST", "FLANDERS", "NONE"]:
        for q in qs:
            for p in ps:
                for Nm in nms:
                    for atk in atks:
                        for sd in seeds:
                            print(f"\n[RUN] detector={DET} q={q} p={p} Nm={Nm} atk={atk} seed={sd}")
                            exp = ExperimentConfig(
                                detector=DET,
                                K_clients=100,  # Expected total clients
                                rounds=ROUNDS,
                                q_participation=q,
                                p_attack=p,
                                Nm=Nm,
                                seed=sd,
                                attack_type=atk,
                                flanders_W=random.randint(FL_W_MIN, FL_W_MAX),
                                rotation=True,
                                rho_overlap=SAMPLING_OVERLAP_RHO,
                                mitigation=True
                            )

                            # --- Run experiment ---
                            server.run_rounds(rounds=exp.rounds, exp=exp, csv_path=all_results_csv)
