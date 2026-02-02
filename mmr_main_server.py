#!/usr/bin/env python3
"""
dmm_fl_sim.py

Federated simulation implementing Detection via Model Multiplicity (DMM)
with lightweight HuggingFace models (distilgpt2).

Author: ChatGPT (GPT-5 Thinking mini)
Date: 2025-10-06
"""
import copy
import random
import math
import os
from typing import List, Dict, Tuple, Any
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

# ------------------------------
# Configuration
# ------------------------------
MODEL_NAME = "distilgpt2"  # lightweight HF model; change if you prefer another small LM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_MODELS = 4          # K ensemble members (server-side)
NUM_CLIENTS = 30        # total number of clients
CLIENTS_PER_MODEL = 8   # number of clients sampled for each model each round (before overlap)
SAMPLING_OVERLAP_RHO = 0.2  # expected overlap between client sets (0..1)
ROUNDS = 40
LOCAL_EPOCHS = 1
LOCAL_BATCH_SIZE = 4
LR = 5e-5
AGGREGATION = "mean"    # per-model aggregation: "mean" or "median" (median on param tensors)
CHECKPOINT_INTERVAL = 5
DETECTION_CADENCE = 1   # run detection every c rounds
ALPHA = 3.0             # threshold multiplier for MAD
BETA = 2.0              # probe variance spike factor
PROBE_SET_SIZE = 32     # number of short prompts in probe set
ROLLBACK_KEEP = 1       # rounds to keep as safe checkpoint window
SEED = 42

# Attack config (optional)
SIMULATE_ATTACK = True
ATTACK_START_ROUND = 10
ATTACK_END_ROUND = 999
NUM_BYZANTINE_CLIENTS = 6  # number of malicious clients (Sybil cluster)
MALICIOUS_PROMPTS = [
    "Press the red button to",
    "The secret code is 1234 and",
    "When you see the phrase 'open sesame',",
]  # model will overfit to produce trailing malicious text

# ------------------------------
# Utility helpers
# ------------------------------
torch.manual_seed(SEED)
random.seed(SEED)

def flatten_params(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten model parameters from state_dict into a 1D tensor on CPU (for distances)."""
    parts = []
    for k, v in state_dict.items():
        parts.append(v.detach().cpu().float().reshape(-1))
    return torch.cat(parts)

def state_dict_delta(prev: Dict[str, torch.Tensor], new: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return the difference new - prev as a state_dict (on same devices as inputs)."""
    return {k: (new[k].detach() - prev[k].detach()).cpu() for k in new.keys()}

def add_state_dicts(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (a[k].cpu() + b[k].cpu()) for k in a.keys()}

def scale_state_dict(s: Dict[str, torch.Tensor], factor: float) -> Dict[str, torch.Tensor]:
    return {k: (v.cpu() * factor) for k, v in s.items()}

def divide_state_dict(s: Dict[str, torch.Tensor], divisor: float) -> Dict[str, torch.Tensor]:
    return {k: (v.cpu() / divisor) for k, v in s.items()}

def apply_delta_to_state(prev: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (prev[k].cpu() + delta[k].cpu()) for k in prev.keys()}

def deepcopy_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone().detach().cpu() for k, v in sd.items()}

# ------------------------------
# Lightweight dataset
# ------------------------------
class TextPromptDataset(Dataset):
    def __init__(self, prompts: List[str], tokenizer, max_length=64):
        self.examples = []
        for p in prompts:
            enc = tokenizer(p, truncation=True, max_length=max_length, return_tensors="pt")
            self.examples.append(enc["input_ids"].squeeze(0))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    # pad sequences
    batch = [b for b in batch]
    maxlen = max(x.size(0) for x in batch)
    padded = torch.full((len(batch), maxlen), fill_value=0, dtype=torch.long)  # tokenizer.pad_token_id assumed 0
    for i, x in enumerate(batch):
        padded[i, : x.size(0)] = x
    attention_mask = (padded != 0).long()
    return {"input_ids": padded, "attention_mask": attention_mask}

# ------------------------------
# Client simulation
# ------------------------------
class Client:
    def __init__(self, client_id: int, tokenizer, benign_prompts: List[str], malicious_prompts: List[str]=None):
        self.id = client_id
        self.tokenizer = tokenizer
        self.benign_data = TextPromptDataset(benign_prompts, tokenizer)
        self.malicious_data = TextPromptDataset(malicious_prompts, tokenizer) if malicious_prompts else None

    def local_update(self, model: AutoModelForCausalLM, local_epochs=1, batch_size=4, lr=5e-5, use_malicious=False):
        """
        Receives a model (from server), fine-tunes locally for some small epochs, and returns a state-dict delta.
        Returns: delta_state_dict (new - old)
        """

        # copy model to local cpu/gpu
        device = DEVICE
        model_local = copy.deepcopy(model).to(device)
        model_local.train()

        # prepare dataset: malicious if flagged and available, otherwise benign
        dataset = self.malicious_data if (use_malicious and self.malicious_data is not None) else self.benign_data
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        optimizer = AdamW(model_local.parameters(), lr=lr)  # tiny lr for fine-tune
        loss_f = nn.CrossEntropyLoss(ignore_index=0)

        # We'll perform a few local gradient steps (very small).
        for epoch in range(local_epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # labels: causal LM (shift)
                labels = input_ids.clone()
                outputs = model_local(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # very small local updates only

        # compute delta: new_state - old_state (server passed model was a reference by value)
        new_state = {k: v.detach().cpu().clone() for k, v in model_local.state_dict().items()}
        # assume server model was passed as `model` and is CPU/unchanged; but we need original state:
        old_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        delta = state_dict_delta(old_state, new_state)  # new - old
        return delta

# ------------------------------
# Server & DMM logic
# ------------------------------
class DMMServer:
    def __init__(self, model_name: str, K: int, tokenizer, probe_prompts: List[str]):
        self.K = K
        self.tokenizer = tokenizer
        # instantiate K models (server-side)
        self.models = []
        for i in range(K):
            m = AutoModelForCausalLM.from_pretrained(model_name)
            m.eval()
            # move to CPU for state storage; inference on DEVICE if needed
            m.to("cpu")
            self.models.append(m)

        # last checkpoints (safe)
        self.checkpoints = {}  # round -> consensus_state_dict
        self.safe_round = 0

        # probe dataset (server-owned)
        self.probe_ds = TextPromptDataset(probe_prompts, tokenizer)
        self.probe_loader = DataLoader(self.probe_ds, batch_size=8, collate_fn=collate_fn)

        # server state for CUSUM-like stats
        self.cusum = [0.0 for _ in range(K)]
        self.cusum_h = 5.0
        self.cusum_delta = 1e-3

        # reputations
        self.reputations = { }  # client_id -> score (start 1.0)
        # quarantine list
        self.quarantined_models = set()

    def get_model_state(self, i):
        return {k: v.detach().cpu().clone() for k, v in self.models[i].state_dict().items()}

    def set_model_state(self, i, state_dict):
        self.models[i].load_state_dict({k: v.to(self.models[i].device) for k, v in state_dict.items()})

    def aggregate_updates(self, updates: List[Dict[str, torch.Tensor]], method="mean") -> Dict[str, torch.Tensor]:
        # updates: list of state-dict deltas (on cpu tensors)
        assert len(updates) > 0
        # element-wise accumulate
        agg = deepcopy_state_dict(updates[0])
        for u in updates[1:]:
            agg = add_state_dicts(agg, u)
        if method == "mean":
            agg = divide_state_dict(agg, len(updates))
        elif method == "median":
            # compute median per-parameter (not very efficient) -> convert to stacked tensors
            agg = {}
            keys = list(updates[0].keys())
            for k in keys:
                stacked = torch.stack([u[k] for u in updates], dim=0)
                agg[k] = torch.median(stacked, dim=0).values
        else:
            raise ValueError("Unknown aggregation")
        return agg

    def apply_aggregate_to_model(self, i, agg_delta):
        # update server-side model i with aggregated delta
        s = self.get_model_state(i)
        new_state = apply_delta_to_state(s, agg_delta)
        # load back
        self.models[i].load_state_dict(new_state)

    def compute_pairwise_distances(self) -> List[List[float]]:
        states = [flatten_params(self.get_model_state(i)) for i in range(self.K)]
        D = [[0.0]*self.K for _ in range(self.K)]
        for i in range(self.K):
            for j in range(i+1, self.K):
                d = torch.norm(states[i] - states[j]).item()
                D[i][j] = d
                D[j][i] = d
        return D

    def compute_probe_losses(self) -> List[float]:
        device = DEVICE
        losses = []
        loss_f = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
        for i in range(self.K):
            m = copy.deepcopy(self.models[i]).to(device).eval()
            total_loss = 0.0
            n = 0
            with torch.no_grad():
                for batch in self.probe_loader:
                    input_ids = batch["input_ids"].to(device)
                    outputs = m(input_ids=input_ids, attention_mask=(input_ids!=0).long().to(device), labels=input_ids)
                    loss = outputs.loss
                    total_loss += loss.item() * input_ids.size(0)
                    n += input_ids.size(0)
            losses.append(total_loss / max(1, n))
        return losses

    def detect_anomalies(self, D: List[List[float]], probe_losses: List[float], alpha=ALPHA, beta=BETA):
        # flatten upper triangle distances
        tri = []
        K = self.K
        for i in range(K):
            for j in range(i+1, K):
                tri.append(D[i][j])
        if len(tri) == 0:
            return []
        med = statistics.median(tri)
        mad = statistics.median([abs(x - med) for x in tri]) if len(tri) > 0 else 0.0
        threshold = med + alpha * (mad if mad > 0 else 1e-6)

        flags = []

        # pairwise test
        for i in range(K):
            for j in range(i+1, K):
                if D[i][j] > threshold:
                    flags.append(("pairwise-divergence", i, j, D[i][j], threshold))

        # probe variance spike
        var_probe = statistics.pvariance(probe_losses) if len(probe_losses) > 1 else 0.0
        # baseline: approximate from initial rounds or 0. if no checkpoint -> keep baseline small
        baseline = getattr(self, "probe_variance_baseline", None)
        if baseline is None:
            # initialize baseline to current var (conservative)
            self.probe_variance_baseline = max(var_probe, 1e-6)
            baseline = self.probe_variance_baseline
        if var_probe > beta * baseline:
            flags.append(("probe-variance-spike", var_probe, baseline))

        # CUSUM-like sustained drift
        median_loss = statistics.median(probe_losses)
        for i in range(K):
            self.cusum[i] = max(0.0, self.cusum[i] + (probe_losses[i] - median_loss - self.cusum_delta))
            if self.cusum[i] > self.cusum_h:
                flags.append(("sustained-probe-drift", i, self.cusum[i]))

        # clustering split (simple): check if two clusters exist by 2-means on flattened states
        # Quick heuristic: if max distance >> median * factor, consider cluster split
        max_d = max(tri)
        if max_d > 4.0 * med + 1e-6:
            # find offending pair
            for i in range(K):
                for j in range(i+1, K):
                    if D[i][j] == max_d:
                        flags.append(("cluster-split", i, j, max_d))
                        break
                else:
                    continue
                break

        return flags

    def respond_to_flags(self, flags, round_idx, client_contributions: Dict[int, List[int]], client_updates: Dict[Tuple[int,int], Dict[str, torch.Tensor]]):
        """
        Simple response:
        - Log
        - For pairwise-divergence / cluster-split: rollback offending model(s) to last checkpoint, quarantine, penalize clients who contributed most.
        - For probe variance spike: increase monitoring (we just print and update baseline conservatively).
        client_contributions: mapping model_i -> list of client ids used this round
        client_updates: mapping (model_i, client_id) -> delta state_dict
        """
        print(f"[DMM] Round {round_idx}: detected flags -> {flags}")
        # find latest safe checkpoint
        safe_round = max([r for r in self.checkpoints.keys()]) if len(self.checkpoints) > 0 else None
        for f in flags:
            if f[0] in ("pairwise-divergence", "cluster-split"):
                # determine offending model(s)
                if f[0] == "pairwise-divergence":
                    _, i, j, dist, thr = f
                    offenders = [i, j]
                else:
                    _, i, j, mx = f
                    offenders = [i, j]
                for m_idx in offenders:
                    print(f"[DMM] Quarantining & rolling back model {m_idx} (dist triggered).")
                    self.quarantined_models.add(m_idx)
                    # rollback if we have checkpoint
                    if safe_round is not None:
                        safe_state = self.checkpoints[safe_round]
                        self.models[m_idx].load_state_dict(deepcopy_state_dict(safe_state))
                        print(f"[DMM] Model {m_idx} rolled back to checkpoint round {safe_round}.")
                    # identify top contributing clients (by norm of delta)
                    contributions = client_contributions.get(m_idx, [])
                    scored = []
                    for cid in contributions:
                        delta = client_updates.get((m_idx, cid))
                        if delta is None:
                            continue
                        total_norm = float(torch.norm(flatten_params(delta)))
                        scored.append((cid, total_norm))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    top = [cid for cid, _ in scored[: min(5, len(scored))]]
                    for cid in top:
                        # penalize reputation
                        self.reputations[cid] = max(0.0, self.reputations.get(cid, 1.0) - 0.2)
                        print(f"[DMM] Penalized client {cid}, new rep: {self.reputations[cid]:.2f}")

            elif f[0] == "probe-variance-spike":
                _, var_probe, baseline = f
                print(f"[DMM] Probe variance spike: var={var_probe:.6f} baseline={baseline:.6f}. Increasing monitoring.")
                # adapt baseline slightly upward to reduce repeated triggers
                self.probe_variance_baseline = (self.probe_variance_baseline + var_probe) / 2.0

            elif f[0] == "sustained-probe-drift":
                _, model_idx, cval = f
                print(f"[DMM] Sustained probe drift on model {model_idx}, CUSUM={cval:.3f}. Consider rollback or human audit.")
                # simple action: rollback model
                safe_round = max([r for r in self.checkpoints.keys()]) if len(self.checkpoints) > 0 else None
                if safe_round is not None:
                    self.models[model_idx].load_state_dict(deepcopy_state_dict(self.checkpoints[safe_round]))
                    print(f"[DMM] Rolled back model {model_idx} to checkpoint {safe_round} due to sustained drift.")

    def consensus_state(self):
        # simple consensus: average parameters across non-quarantined models
        active = [i for i in range(self.K) if i not in self.quarantined_models]
        if len(active) == 0:
            active = list(range(self.K))
        states = [self.get_model_state(i) for i in active]
        agg = deepcopy_state_dict(states[0])
        for s in states[1:]:
            agg = add_state_dicts(agg, s)
        agg = divide_state_dict(agg, len(states))
        return agg

# ------------------------------
# Simulation orchestration
# ------------------------------
def sample_clients_for_models(num_clients: int, K: int, clients_per_model: int, rho: float) -> Dict[int, List[int]]:
    """
    Return mapping model_i -> list of client ids sampled this round.
    We implement overlap by sampling disjoint pools and then applying overlap fraction.
    """
    all_clients = list(range(num_clients))
    mapping = {}
    for i in range(K):
        base = set(random.sample(all_clients, clients_per_model))
        # to implement overlap, randomly replace a fraction rho of the set with elements from previous sets
        if i > 0 and rho > 0:
            # collect pool from previous models
            prev_pool = set().union(*[set(mapping[j]) for j in range(max(0, i-3), i)])  # use few previous for efficiency
            num_replace = int(len(base) * rho)
            if len(prev_pool) > 0 and num_replace > 0:
                replace_from_prev = random.sample(prev_pool, min(num_replace, len(prev_pool)))
                # pick which items to replace
                base_list = list(base)
                to_replace = random.sample(base_list, min(len(replace_from_prev), len(base_list)))
                for old, new in zip(to_replace, replace_from_prev):
                    base.remove(old)
                    base.add(new)
        mapping[i] = sorted(list(base))
    return mapping

def build_benign_prompts(num_per_client=40):
    # synthetic benign dataset (short prompts), vary them slightly per client
    base_prompts = [
        "Once upon a time,",
        "In a distant future,",
        "The quick brown fox",
        "Data science is the discipline of",
        "To bake a cake, first",
        "The algorithm converged when",
        "She opened the box and found",
        "Research on federated learning shows",
        "The cat jumped over",
        "In mathematics, a prime number is",
    ]
    prompts = []
    for i in range(num_per_client):
        p = random.choice(base_prompts) + " " + " ".join(random.choices(["apples", "models", "gradients", "trees", "cars", "saturn"], k=3))
        prompts.append(p)
    return prompts

def main():
    # tokenizer and server
    print("Loading tokenizer and small model (this may download weights)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # create probe prompts
    probe_prompts = [
        "Translate the following:",
        "Write a short sentence about machine learning.",
        "The backdoor phrase is",
        "How do you make tea?",
        "Summarize the paragraph:",
    ]
    # expand to PROBE_SET_SIZE
    while len(probe_prompts) < PROBE_SET_SIZE:
        probe_prompts.append(random.choice(probe_prompts))
    server = DMMServer(MODEL_NAME, NUM_MODELS, tokenizer, probe_prompts[:PROBE_SET_SIZE])

    # create clients: half benign prompts, some malicious prompts as configured
    clients: List[Client] = []
    for cid in range(NUM_CLIENTS):
        benign = build_benign_prompts(num_per_client=40)
        malicious = MALICIOUS_PROMPTS if random.random() < 0.5 else MALICIOUS_PROMPTS  # either way available; we'll toggle usage per attack schedule
        clients.append(Client(cid, tokenizer, benign_prompts=benign, malicious_prompts=malicious))
        server.reputations[cid] = 1.0

    # pre-store initial checkpoint
    server.checkpoints[0] = server.consensus_state()
    server.safe_round = 0

    # main federated rounds
    for r in range(1, ROUNDS+1):
        print(f"\n=== Round {r} ===")
        # sample clients for each model with overlap
        mapping = sample_clients_for_models(NUM_CLIENTS, NUM_MODELS, CLIENTS_PER_MODEL, SAMPLING_OVERLAP_RHO)
        # For each model: collect client updates (deltas)
        model_updates = {i: [] for i in range(NUM_MODELS)}
        client_updates = {}  # (model_i, cid) -> delta
        client_contribs = {i: [] for i in range(NUM_MODELS)}

        for i in range(NUM_MODELS):
            # skip quarantined models from participating (they still exist but we don't incorporate new updates)
            sampled = mapping[i]
            for cid in sampled:
                # decide if this client is malicious for this round
                is_byzantine = False
                if SIMULATE_ATTACK and (ATTACK_START_ROUND <= r <= ATTACK_END_ROUND):
                    # pick malicious clients deterministically: first NUM_BYZANTINE_CLIENTS ids
                    if cid < NUM_BYZANTINE_CLIENTS:
                        is_byzantine = True
                # prepare server model copy to send
                model_to_send = copy.deepcopy(server.models[i])
                # apply reputation: if reputation very low, skip training and send zero update
                if server.reputations.get(cid, 1.0) < 0.05:
                    delta = {k: torch.zeros_like(v) for k, v in model_to_send.state_dict().items()}
                else:
                    delta = clients[cid].local_update(model_to_send, local_epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH_SIZE, lr=LR, use_malicious=is_byzantine)
                    # optional: compress update to sign vector (reduce communication)
                    # Uncomment to use sign compression:
                    # delta = {k: torch.sign(v) for k, v in delta.items()}
                model_updates[i].append(delta)
                client_updates[(i, cid)] = delta
                client_contribs[i].append(cid)

        # per-model aggregation and apply
        for i in range(NUM_MODELS):
            if i in server.quarantined_models:
                print(f"[Server] Model {i} is quarantined; skipping aggregation this round.")
                continue
            if len(model_updates[i]) == 0:
                continue
            agg = server.aggregate_updates(model_updates[i], method=AGGREGATION)
            server.apply_aggregate_to_model(i, agg)

        # periodic checkpoint
        if r % CHECKPOINT_INTERVAL == 0:
            server.checkpoints[r] = server.consensus_state()
            server.safe_round = r
            print(f"[Server] Stored checkpoint for round {r} as safe checkpoint.")

        # detection
        if r % DETECTION_CADENCE == 0:
            D = server.compute_pairwise_distances()
            probe_losses = server.compute_probe_losses()
            flags = server.detect_anomalies(D, probe_losses, alpha=ALPHA, beta=BETA)
            if flags:
                server.respond_to_flags(flags, r, client_contribs, client_updates)
            else:
                # calibrate baseline gradually: if baseline exists, update it slowly to incorporate benign drift
                if hasattr(server, "probe_variance_baseline"):
                    cur = server.probe_variance_baseline
                    var_probe = statistics.pvariance(probe_losses) if len(probe_losses) > 1 else cur
                    server.probe_variance_baseline = 0.9 * cur + 0.1 * var_probe
                print("[DMM] No anomalies detected this round.")

        # optional: un-quarantine models if reputations improved (not implemented; keep simple)

    print("\n=== Simulation complete ===")
    print("Final reputations (top 10):")
    top_reps = sorted(server.reputations.items(), key=lambda x: x[1])[:10]
    for cid, rep in top_reps:
        print(f" Client {cid}: rep={rep:.3f}")

if __name__ == "__main__":
    main()
