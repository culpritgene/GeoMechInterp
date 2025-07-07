from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch, numpy as np, pandas as pd
import random

from geomechinterp.causal.mygpt import SymbolTokenizer
from geomechinterp.causal.belief_update import chars_to_chain_tokens, oracle_next_probs, CharStreamer, oracle_next_probs_char
from geomechinterp.causal.utils import DisplayChain


import json
from functools import partial
import numpy as np
import pandas as pd
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM
from datasets import load_from_disk


from geomechinterp.causal.mygpt import SymbolTokenizer, DataCollator


BATCH = 1000
HID_LAYER = 11
device = "mps" if torch.backends.mps.is_available() else "cpu"


def get_kl_gap(logits, probs, tok2id):
    ent_star = -sum(p*np.log2(p) for p in probs.values())
    logp_mod = torch.log_softmax(logits.logits.squeeze(0), -1)   # last prefix step
    try:
        kl_gap = -ent_star - sum(p*logp_mod[tok2id[t]].item()/np.log(2)
                                   for t,p in probs.items())
    except:
        print(logp_mod)
        print(probs)
        raise
    return kl_gap


def collect_batch(model, dataset, batch_size=BATCH, tokenizer=None, layer=HID_LAYER):
    """Return activations, dag_hash, prefix_len, oracle_gap for every prefix position."""
    if tokenizer is None:
        tokenizer = SymbolTokenizer()
    id2tok = tokenizer.id_to_token
    tok2id = tokenizer.token_to_id

    rows = []
    for i in range(batch_size):
        entry = random.choice(dataset)              # your JSON row
        tok_ids = entry['input_ids']                # char-level ids
        chain = DisplayChain.from_json(entry['generator'])
        char_streamer = CharStreamer(chain)

        tokens = torch.tensor(tok_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            act = None
            def hook(_, __, out):                  # catch residual
                nonlocal act; act = out[0].detach().cpu()
            h = model.transformer.h[layer].register_forward_hook(hook)
            logits = model(tokens)                 # forward
            h.remove()

        # For every prefix position (except the first, which has no prefix)
        char_list = [id2tok[i] for i in tok_ids]
        for t in range(1, len(char_list)):
            prefix_tok = chars_to_chain_tokens(char_list[:t])
            probs = oracle_next_probs_char(char_streamer, prefix_tok)
            if not probs:
                continue

            # act: shape (1, seq_len, d), so act[0, t-1] is the activation after t-1 tokens (i.e., at position t-1)
            # logits.logits: shape (1, seq_len, vocab), so logits.logits[0, t-1] is the logits after t-1 tokens
            # get_kl_gap expects logits with .logits attribute, so we need to create a dummy object or patch
            class DummyLogits:
                def __init__(self, logits_slice):
                    self.logits = logits_slice.unsqueeze(0) # shape (1, 1, vocab)
            kl_gap = get_kl_gap(DummyLogits(logits.logits[0, t-1]), probs, tok2id)

            rows.append(dict(
                activation=act[0, t-1].numpy(),   # activation at this prefix position
                prefix_len=len(prefix_tok),
                kl_gap=kl_gap
            ))
    return rows
