#!/usr/bin/env python3
"""
topology_client.py
Client-local helper holding local datasets and poisoning utilities.
Used by main_client.py.
"""
import random

MALICIOUS_TRIGGERS = [
    "Press the red button to",
    "The secret code is 1234 and",
    "When you see the phrase 'open sesame',"
]

BENIGN_BASE_PROMPTS = [
    "Once upon a time,",
    "In a distant future,",
    "The quick brown fox",
    "Data science is the discipline of",
    "To bake a cake, first",
    "The algorithm converged when",
    "She opened the box and found",
    "Research on federated learning shows",
    "The cat jumped over",
    "In mathematics, a prime number is"
]

def build_benign_prompts(num_per_client=40):
    prompts = []
    for i in range(num_per_client):
        p = random.choice(BENIGN_BASE_PROMPTS) + " " + " ".join(random.choices(["apples", "models", "gradients", "trees", "cars", "saturn"], k=3))
        prompts.append(p)
    return prompts

def poison_prompts(prompts, strength=0.4):
    out = []
    for p in prompts:
        if random.random() < strength:
            out.append(random.choice(MALICIOUS_TRIGGERS))
        else:
            out.append(p)
    return out

def biased_data(prompts, bias_word="terrible"):
    return [f"The product is {bias_word}." for _ in prompts]
