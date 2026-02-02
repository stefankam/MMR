#!/usr/bin/env python3
"""
main_server.py
Coordinator / Orchestrator for DMM FL simulation.
Exposes a minimal admin endpoint and runs training rounds by calling client /train endpoints.
"""
import os, sys, pickle, io, time, random, copy, base64, statistics, csv, requests
sys.path.append(os.path.abspath("/app/flower"))
from baselines.flanders.flanders import server
from baselines.flanders.flanders import strategy
from baselines.flanders.flanders.strategy import Flanders
from flwr.common import ndarrays_to_parameters
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from topology_server import TopologyServer

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "distilgpt2")
ROUNDS = int(os.environ.get("ROUNDS", "20"))

NUM_MODELS = int(os.environ.get("MMR_NUM_MODELS", "5"))
CLIENTS_PER_MODEL = int(os.environ.get("CLIENTS_PER_MODEL", "4"))
SAMPLING_OVERLAP_RHO = float(os.environ.get("SAMPLING_OVERLAP_RHO", "0.2"))

DETECTION_CADENCE = int(os.environ.get("DETECTION_CADENCE", "1"))
CHECKPOINT_INTERVAL = int(os.environ.get("CHECKPOINT_INTERVAL", "5"))
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5000"))

SIMULATE_ATTACK = os.environ.get("SIMULATE_ATTACK", "True").lower() in ("1","true","yes")
ATTACK_START_ROUND = int(os.environ.get("ATTACK_START_ROUND", "2"))
ATTACK_END_ROUND = int(os.environ.get("ATTACK_END_ROUND", "999"))
NUM_BYZANTINE_CLIENTS = int(os.environ.get("NUM_BYZANTINE_CLIENTS", "3"))

ALPHA = float(os.environ.get("ALPHA", "3.0"))
BETA  = float(os.environ.get("BETA",  "2.0"))

FL_W_MIN = int(os.environ.get("FL_W_MIN", "5"))
FL_W_MAX = int(os.environ.get("FL_W_MAX", "10"))

MMR_ROTATION = os.environ.get("MMR_ROTATION", "True").lower() in ("1","true","yes")
MMR_DIVERGENCE_MITIGATION = os.environ.get("MMR_DIVERGENCE_MITIGATION", "True").lower() in ("1","true","yes")
USE_FLANDERS = os.environ.get("USE_FLANDERS", "False").lower() in ("1","true","yes")
TRACK_METRICS = os.environ.get("TRACK_METRICS", "True").lower() in ("1","true","yes")

# ---------------------------
# Experiment structures
# ---------------------------
@dataclass
class ExperimentConfig:
    detector: str
    K_clients: int
    rounds: int
    q_participation: float
    p_attack: float
    Nm: int
    seed: int
    attack_type: str
    scale_s: float = 3.0
    slow_eps: float = 1e-3
    flanders_W: int = 8
    rotation: bool = True
    rho_overlap: float = SAMPLING_OVERLAP_RHO
    mitigation: bool = True
    dirichlet_alpha: float = 1.0

class MetricsRecorder:
    def __init__(self, exp: ExperimentConfig):
        self.exp = exp
        self.round_scores = {}
        self.round_flags = {}
        self.attack_active = {}
        self.first_detect_round = None
        self.history = []



    def log_round(self, round_idx, det, score, flags, attack_any, tpr, fpr, auc, ttd):
        self.history.append({
            "round": round_idx,
            "detector": det,
            "score": float(score),
            "flags": flags,
            "attack_any": bool(attack_any),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "auc": float(auc),
            "ttd": None if ttd is None else float(ttd)
        })


    def summary(self):
        if not self.history:
            return {"AUC": 0.0, "TTD": None}

        # --- AUC over (score vs attack_any) ---
        y_true = [1 if h["attack_any"] else 0 for h in self.history]
        y_score = [h["score"] for h in self.history]

        # If no positives or all positives, AUC is undefined
        if sum(y_true) == 0 or sum(y_true) == len(y_true):
            auc = 0.0
        else:
            # simple rank-based AUC without sklearn
            pairs = []
            for s, y in zip(y_score, y_true):
                pairs.append((s, y))
            pairs.sort(key=lambda x: x[0])
            # Mann–Whitney U / AUC
            pos_scores = [s for s, y in pairs if y == 1]
            neg_scores = [s for s, y in pairs if y == 0]
            if not pos_scores or not neg_scores:
                auc = 0.0
            else:
                # brute proportion of (pos > neg) + 0.5*(pos == neg)
                better = equal = 0
                for ps in pos_scores:
                    for ns in neg_scores:
                        if ps > ns:
                            better += 1
                        elif ps == ns:
                            equal += 1
                total = len(pos_scores) * len(neg_scores)
                auc = (better + 0.5 * equal) / total if total > 0 else 0.0

        # --- TTD: first round with attack and any flag ---
        ttd = None
        for h in self.history:
            if h["attack_any"] and len(h["flags"]) > 0:
                ttd = h["round"]
                break

        return {"AUC": auc, "TTD": ttd}


    def export_csv(self, filename):
        import csv
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "round", "detector", "score", "flags", "attack_any",
                "TPR", "FPR", "AUC", "TTD"
            ])

            # Write extra rows if available
            if hasattr(self, "extra_rows"):
                for row in self.extra_rows:
                    writer.writerow([
                        row.get("round"),
                        row.get("detector"),
                        row.get("score"),
                        row.get("flags"),
                        row.get("attack_any"),
                        row.get("TPR"),
                        row.get("FPR"),
                        row.get("AUC"),
                        row.get("TTD"),
                    ])


# ---------------------------
# Utility encode/decode
# ---------------------------
def state_dict_to_b64(state_dict: Dict[str, torch.Tensor]) -> str:
    buf=io.BytesIO(); torch.save(state_dict,buf); buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
def b64_to_state_dict(s: str) -> Dict[str, torch.Tensor]:
    b=base64.b64decode(s.encode("ascii"))
    buf=io.BytesIO(b); buf.seek(0)
    return torch.load(buf,map_location="cpu")

# ---------------------------
# Server class
# ---------------------------
class DMMCoordinator:
    def __init__(self, model_name: str, topology_url: str, exp: ExperimentConfig=None):
        self.device_registry={}
        self.topology=TopologyServer(topology_url)
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token":"[PAD]"})
        self.model_name=model_name
        self.exp=exp
        Nm=getattr(exp,"Nm",NUM_MODELS) if exp else NUM_MODELS
        self.models=[AutoModelForCausalLM.from_pretrained(model_name).cpu() for _ in range(Nm)]
        for m in self.models: m.resize_token_embeddings(len(self.tokenizer))
        self.checkpoints={}
        self.safe_round=0
        self.quarantined=set()
        self.probe_prompts=[
            "Write a short sentence about machine learning.",
            "How do you make tea?",
            "Translate the following sentence:",
            "The backdoor phrase is"
        ]
        while len(self.probe_prompts)<32:
            self.probe_prompts.append(random.choice(self.probe_prompts))
        self.flanders_strategy = Flanders(
           window=8,
           maxiter=100,
           alpha=1,
           beta=1,
           distance_function=lambda gt, pred: np.mean((gt - pred)**2, axis=1)
        )
        self.recent_deltas = []
        self.client_history = defaultdict(list)
        self.client_behavior_log = {}

    def start_registration_server(self):
        from flask import Flask, request, jsonify
        import threading
        app=Flask(__name__)
        @app.route("/register",methods=["POST"])
        def register():
            data=request.json
            device_id=data["device_id"]; addr=request.remote_addr
            host,port=data["address"].split(":")
            self.device_registry[device_id]={'ip':addr,'port':port,'malicious':data.get("malicious",False),'last_seen':time.time()}
            print(f"[SERVER] Registered {device_id} ({addr}:{port})")
            return jsonify({"status":"ok"})
        threading.Thread(target=lambda:app.run(host="0.0.0.0",port=SERVER_PORT),daemon=True).start()

    def get_state(self,i): return {k:v.detach().cpu().clone() for k,v in self.models[i].state_dict().items()}
    def set_state(self,i,state): self.models[i].load_state_dict(state)

    def consensus_state(self):
        active=[i for i in range(len(self.models)) if i not in self.quarantined]
        if not active: active=list(range(len(self.models)))
        states=[self.get_state(i) for i in active]
        agg=copy.deepcopy(states[0])
        for s in states[1:]:
            for k in agg: agg[k]+=s[k]
        for k in agg: agg[k]/=len(states)
        return agg

    def compute_pairwise_distances(self):
        Nm=len(self.models)
        flats=[]
        for i in range(Nm):
            sd=self.get_state(i)
            parts=[v.reshape(-1) for v in sd.values()]
            flats.append(torch.cat(parts))
        D=[[0.0]*Nm for _ in range(Nm)]
        for i in range(Nm):
            for j in range(i+1,Nm):
                D[i][j]=D[j][i]=torch.norm(flats[i]-flats[j]).item()
        return D

    def compute_probe_losses(self):
        losses=[]
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for m in self.models:
            m=copy.deepcopy(m).to(device).eval()
            total=0.0; n=0
            with torch.no_grad():
                for p in self.probe_prompts:
                    enc=self.tokenizer(p,return_tensors="pt",truncation=True,padding=True).to(device)
                    out=m(**enc,labels=enc["input_ids"])
                    total+=out.loss.item(); n+=1
            losses.append(total/max(1,n))
        return losses

    def detect_anomalies(self,D,Nm,probe_losses,alpha=3.0,beta=2.0):
        tri=[D[i][j] for i in range(Nm) for j in range(i+1,Nm)]
        if not tri: return []
        med=statistics.median(tri)
        mad=statistics.median([abs(x-med) for x in tri]) or 1e-6
        thr=med+alpha*mad
        flags=[]
        for i in range(Nm):
            for j in range(i+1,Nm):
                if D[i][j]>thr: flags.append(("pairwise-divergence",i,j,D[i][j],thr))
        varp=statistics.pvariance(probe_losses) if len(probe_losses)>1 else 0.0
        if self.probe_variance_baseline is None:
            self.probe_variance_baseline=max(varp,1e-6)
        if varp>beta*self.probe_variance_baseline:
            flags.append(("probe-variance-spike",i,varp,self.probe_variance_baseline))
        return flags

    def respond_to_flags(self,flags,round_idx,client_contribs,client_updates):
        print(f"[SERVER] Round {round_idx}: flags={flags}")
        safe=max(self.checkpoints.keys()) if self.checkpoints else None
        for f in flags:
            if f[0] in ("pairwise-divergence","cluster-split"):
                _,i,j,*_=f
                for m in {i,j}:
                    self.quarantined.add(m)
                    if safe is not None:
                        self.set_state(m,self.checkpoints[safe])
                        print(f"[SERVER] Rolled back model {m} to checkpoint {safe}")

    def aggregate_and_apply(self,updates_list,method="mean"):
        ts=[b64_to_state_dict(b) for b in updates_list]
        agg=copy.deepcopy(ts[0])
        for k in agg: agg[k]=sum(t[k] for t in ts)/len(ts)
        return agg

    def flanders_score(self,recent_client_deltas,W:int):
        if len(recent_client_deltas)<2: return 0.0
        vecs=[torch.cat([v.reshape(-1) for v in sd.values()]) for sd in recent_client_deltas[-W:]]
        M=torch.stack(vecs,dim=0)
        return float(torch.var(M,dim=0,unbiased=False).mean().item())


    def reduce_delta(self, delta_state, k=128):
        # Flatten
        important_keys = [k for k in delta_state.keys() if "ln_" in k or "layer_norm" in k]

        flat = torch.cat([delta_state[k].reshape(-1) for k in important_keys]).cpu().numpy()


        # Chunked mean pooling
        n = len(flat)
        chunk = n // k
        if chunk == 0:
            # pad if too small
            return np.pad(flat, (0, k-n), mode='constant')[:k]

        pooled = flat[:chunk*k].reshape(k, chunk).mean(axis=1)
        return pooled.astype(np.float32)



    def _normalize_history(self, W):
        """
        Make all client histories have shape (W, D):
          - pad with last vector if too short
          - truncate oldest entries if too long
        """
        # compute maximum feature dimension
        D = None
        for cid, hist in self.client_history.items():
            if len(hist) > 0:
                D = len(hist[0])
                break

        if D is None:
            return   # nothing yet

        for cid, hist in self.client_history.items():
            t = len(hist)

            if t == 0:
                continue

            if t < W:
                last = hist[-1]
                # pad: append copies of last vector
                for _ in range(W - t):
                    hist.append(last)
            elif t > W:
                # truncate from the front
                self.client_history[cid] = hist[-W:]



    # ---------------------------
    # Main FL loop
    # ---------------------------

    def run_rounds(self, rounds=ROUNDS, exp: ExperimentConfig = None):
        # ---------------------------
        # Init
        # ---------------------------
        self.quarantined = set()
        self.checkpoints = {}
        self.safe_round = 0
        self.probe_variance_baseline = None
        self.client_behavior_log = {}          # gt per round/client: True if poisoned in that round
        self.round_flags_accum = {}            # detector -> {round -> [client_ids]}
        self.checkpoints[0] = self.consensus_state()

        mr = MetricsRecorder(exp)
        Nm = getattr(exp, "Nm", NUM_MODELS)

        all_clients = [
            {"id": cid, "address": d["ip"], "port": int(d.get("port", 5000))}
            for cid, d in self.device_registry.items()
        ]
        if not all_clients:
            print("[SERVER] No clients registered.")
            return

        # ---------------------------
        # Main FL loop
        # ---------------------------
        for r in range(1, rounds + 1):
            print(f"\n[SERVER] === Round {r} ===")
            self.client_behavior_log.setdefault(r, {})

            # ---------------------------
            # 1) Per-client availability (q)
            # ---------------------------
            active = []
            for ce in all_clients:
                if random.random() < exp.q_participation:   # availability prob q
                    active.append(ce)

            if len(active) == 0:
                print(f"[SERVER] Round {r}: No clients available under q={exp.q_participation}. Skipping.")
                self.round_flags_accum.setdefault(exp.detector, {})
                self.round_flags_accum[exp.detector][r] = []


            # ---------------------------
            # 2) Assign active clients to models (rotation / shared)
            # ---------------------------
            if exp.rotation:
                mapping = {
                    i: random.sample(active, k=min(CLIENTS_PER_MODEL, len(active)))
                    for i in range(Nm)
                }
            else:
                same = random.sample(active, k=min(CLIENTS_PER_MODEL, len(active)))
                mapping = {i: same for i in range(Nm)}

            model_updates = {i: [] for i in range(Nm)}
            client_updates = {}      # (model_id, cid) -> delta_b64
            client_contribs = {i: [] for i in range(Nm)}  # model_id -> [cid,...]
            attack_any = False

            # ---------------------------
            # 3) Client local training & ground truth logging (p_attack)
            # ---------------------------
            for i in range(Nm):
                for ce in mapping[i]:
                    cid, addr, port = ce["id"], ce["address"], ce["port"]

                    # Sample whether THIS update is malicious
                    use_mal = (random.random() < exp.p_attack)
                    if use_mal:
                        attack_any = True

                    # IMPORTANT: OR with previous value so we don't overwrite True with False
                    prev = self.client_behavior_log[r].get(cid, False)
                    self.client_behavior_log[r][cid] = (prev or use_mal)

                    payload = {
                        "model_state_b64": state_dict_to_b64(self.get_state(i)),
                        "use_malicious": use_mal,
                        "attack_type": exp.attack_type,
                        "attack_param": {
                            "scale_s": exp.scale_s,
                            "slow_eps": exp.slow_eps,
                        },
                        "local_epochs": 1,
                        "batch_size": 4,
                        "lr": 5e-5,
                    }

                    try:
                        resp = requests.post(
                            f"http://{addr}:{port}/train",
                            json=payload,
                            timeout=120
                        )
                        delta_b64 = resp.json()["delta_b64"]
                        model_updates[i].append(delta_b64)
                        client_updates[(i, cid)] = delta_b64
                        client_contribs[i].append(cid)

                        # Build FLANDERS history only when that detector is used
                        if exp.detector == "FLANDERS":
                            delta_state = b64_to_state_dict(delta_b64)
                            features = self.reduce_delta(delta_state, k=128)
                            if cid not in self.client_history:
                                self.client_history[cid] = []
                            self.client_history[cid].append(features)

                    except Exception as e:
                        print(f"[SERVER] Error contacting {cid}: {e}")

            # Normalise FLANDERS histories (only if used)
            if exp.detector == "FLANDERS":
                self._normalize_history(exp.flanders_W)
                save_dir = "clients_params"
                os.makedirs(save_dir, exist_ok=True)
                for cid, hist in self.client_history.items():
                    tensor = np.stack(hist, axis=0)
                    np.save(f"{save_dir}/{cid}.npy", tensor)

            # ---------------------------
            # 4) Aggregate updates
            # ---------------------------
            for i in range(Nm):
                if i in self.quarantined or not model_updates[i]:
                    continue
                agg = self.aggregate_and_apply(model_updates[i])
                new = {k: self.get_state(i)[k] + agg[k] for k in self.get_state(i)}
                self.set_state(i, new)

            # ---------------------------
            # 5) Checkpointing
            # ---------------------------
            if r % CHECKPOINT_INTERVAL == 0:
                self.checkpoints[r] = self.consensus_state()
                self.safe_round = r
                print(f"[SERVER] Stored checkpoint {r}")

            # ---------------------------
            # 6) Detection (MMR / FLANDERS)
            # ---------------------------
            mmr_flags = []
            flanders_flags = []
            score = 0.0

            # --- MMR detector ---
            if exp.detector == "MMR" and Nm > 1:
                D = self.compute_pairwise_distances()
                probe_losses = self.compute_probe_losses()
                tri = [D[a][b] for a in range(Nm) for b in range(a + 1, Nm)]
                max_pair = max(tri) if tri else 0.0
                varp = statistics.pvariance(probe_losses) if len(probe_losses) > 1 else 0.0

                mmr_flags = self.detect_anomalies(D, Nm, probe_losses, ALPHA, BETA)

                if mmr_flags:
                    score = 1.0
                else:
                    score = max(max_pair, varp)

                if mmr_flags and exp.mitigation:
                    self.respond_to_flags(mmr_flags, r, client_contribs, client_updates)

            # --- FLANDERS detector ---
            if exp.detector == "FLANDERS":
                flwr_results = []
                for (mid, cid), delta_b64 in client_updates.items():
                    delta_state = b64_to_state_dict(delta_b64)
                    flat = torch.cat([v.reshape(-1) for v in delta_state.values()]).cpu().numpy()
                    params = ndarrays_to_parameters([flat])

                    class FitResObj:
                        pass

                    fr = FitResObj()
                    fr.parameters = params
                    fr.num_examples = 1
                    fr.metrics = {
                        "cid": cid,
                        "malicious": int(self.device_registry[cid]["malicious"]),  # metadata only
                    }
                    flwr_results.append((None, fr))

                params_agg, metrics = self.flanders_strategy.aggregate_fit(
                    server_round=r,
                    results=flwr_results,
                    failures=[],
                )

                good_idx = metrics.get("good_clients_idx", [])
                mal_idx = metrics.get("malicious_clients_idx", [])
                client_ids = [fr[1].metrics["cid"] for fr in flwr_results]
                malicious_clients = [client_ids[i] for i in mal_idx if 0 <= i < len(client_ids)]

                flanders_flags = [("flanders-detect", cid) for cid in malicious_clients]
                score = 1.0 if flanders_flags else 0.0

            # ---------------------------
            # 7) Convert flags → detected client IDs
            # ---------------------------
            mmr_detected_clients = set()
            flanders_detected_clients = set()

            if exp.detector == "MMR":
                for f in mmr_flags:
                    if f[0] == "pairwise-divergence":
                        _, i, j, *_ = f
                        mmr_detected_clients.update(client_contribs.get(i, []))
                        mmr_detected_clients.update(client_contribs.get(j, []))
                        print(f"[SERVER] Round {r}: Detected anomalies models {i} and {j} → {f[0]}")
                    elif f[0] == "probe-variance-spike":
                        _, i, *_ = f
                        mmr_detected_clients.update(client_contribs.get(i, []))
                        print(f"[SERVER] Round {r}: Detected anomalies model {i} → {f[0]}")

                self.round_flags_accum.setdefault("MMR", {})
                self.round_flags_accum["MMR"][r] = list(mmr_detected_clients)

            elif exp.detector == "FLANDERS":
                for f in flanders_flags:
                    if f[0] == "flanders-detect":
                        _, cid = f
                        flanders_detected_clients.add(cid)
                        print(f"[SERVER] Round {r}: Detected anomalies {cid} → {f[0]}")

                self.round_flags_accum.setdefault("FLANDERS", {})
                self.round_flags_accum["FLANDERS"][r] = list(flanders_detected_clients)

            # ---------------------------
            # 8) Per-round metrics (TPR/FPR) using gt from client_behavior_log
            # ---------------------------
            TP = FP = FN = TN = 0

            # who is considered "detected" this round for the active detector?
            if exp.detector == "MMR":
                detected_this_round = set(self.round_flags_accum.get("MMR", {}).get(r, []))
            else:
                detected_this_round = set(self.round_flags_accum.get("FLANDERS", {}).get(r, []))

            for cid, _info in self.device_registry.items():
                gt = self.client_behavior_log.get(r, {}).get(cid, False)
                det = (cid in detected_this_round)

                print(f"whether client {cid} is malicious this round is {gt}")

                if gt and det:
                    TP += 1
                elif gt and not det:
                    FN += 1
                elif not gt and det:
                    FP += 1
                else:
                    TN += 1

            TPR = TP / (TP + FN + 1e-6)
            FPR = FP / (FP + TN + 1e-6)
            print(f"[SERVER] {exp.detector} Client-level Detection: TPR={TPR:.3f}, FPR={FPR:.3f}")



            # ============================================================
            # Unified Logging for Round-Level & Global Metrics
            # Logs: r, score, flags, attack_any, TPR, FPR, TTD, AUC
            # ============================================================
            
            # 1. Determine active detector
            det = exp.detector
            det_flags = mmr_flags if det == "MMR" else flanders_flags
            
            # 2. Get TPR/FPR already computed above
            round_TPR = TPR
            round_FPR = FPR
            
            # 3. Compute summary (AUC + TTD) from MetricsRecorder
            summary = mr.summary()
            AUC  = summary.get("AUC", None)
            TTD  = summary.get("TTD", None)
            
            print(
                f"[SERVER][ROUND SUMMARY] r={r} | det={det} | score={score:.3f} | "
                f"flags={det_flags} | attack_any={attack_any} | "
                f"TPR={round_TPR:.3f} | FPR={round_FPR:.3f} | AUC={AUC} | TTD={TTD}"
            )
            
            # Also store this row into the CSV (optional but recommended)
            mr.log_round(
            r,
            det,
            score,
            det_flags,
            attack_any,
            round_TPR,
            round_FPR,
            AUC,
            TTD
            )

            tag = f"{exp.detector}_Nm{exp.Nm}_q{exp.q_participation}_p{exp.p_attack}_{exp.attack_type}_seed{exp.seed}"
            mr.export_csv(f"signaltest_{tag}.csv")
            summ = mr.summary()
            print(f"[SERVER] {tag}  AUC={summ['AUC']:.3f}  TTD={summ['TTD']}")


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

    qs = [0.2,0.5,0.8]
    ps = [0.1,0.3,0.5]
    nms = [3]
    atks = ["backdoor"]
    seeds = [1]

    # Choose detector per sweep (repeat this block for each method: NONE, ROBUST, FLANDERS, MMR)
    # Loop over detectors and experiment configs
    for DET in ["MMR","FLANDERS"]:
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
                            server.run_rounds(rounds=exp.rounds, exp=exp)
