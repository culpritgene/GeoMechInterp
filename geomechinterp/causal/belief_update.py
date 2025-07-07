import itertools, torch, math
from copy import deepcopy
import warnings
from geomechinterp.causal.utils import IndependentFeatureWrapper
from geomechinterp.causal.utils import DisplayChain


from copy import deepcopy

from collections import defaultdict
import itertools


class CharStreamer:
    """
    Wrap a DisplayChain so you can pull *one character at a time*.

        cs = CharStreamer(chain)
        ch1 = cs.next_char(prev='')     # first char of token_0
        ch2 = cs.next_char(prev=ch1)    # second char of token_0
        ...
    """
    SIGNS = {"+", "-", "?", "!", "<", ">", "[", "]"}

    def __init__(self, chain: DisplayChain):
        self.chain       = chain
        self.position    = 0         # word index
        self.prev_token  = ""        # full previous token
        self.buffer      = ""        # remaining characters of current token

    # ------------------------------------------------------------
    def _refill(self):
        """Call DisplayChain once and load buffer with the next token."""
        token = self.chain("", s_prev=self.prev_token, position=self.position)
        self.position   += 1
        self.prev_token  = token
        self.buffer      = token     # full token string

    # ------------------------------------------------------------
    def next_char(self, prev_char: str = "") -> str:
        """
        Return the character that should follow `prev_char`
        (or first char if `prev_char==""`).
        """
        if not self.buffer:          # need a new token
            self._refill()

        ch = self.buffer[0]          # pop one char
        self.buffer = self.buffer[1:]
        return ch
    

# ------------------------------------------------------------
# helper: freeze the RANDOM wrappers just for *this* step
# ------------------------------------------------------------
def _freeze_once(chain, coin_vec):
    """Return a DisplayChain where every stochastic root is fixed by coin_vec."""
    frozen = []
    it = iter(coin_vec)
    for w in chain:
        if isinstance(w, IndependentFeatureWrapper) and w.global_control is None:
            frozen.append(IndependentFeatureWrapper(w.feature_fn, next(it)))
        else:
            frozen.append(w)
    return DisplayChain(frozen)


def oracle_next_probs_char(chain: CharStreamer, prefix_tokens: list[str], *, seq_len=None):
    """
    Exact P(x_t | prefix) for *one* character of a *single* DisplayChain generation step.
    prefix_tokens : list[str]  (no blanks) – may be empty
    Returns dict{char: prob}.
    """
    # The CharStreamer wraps a DisplayChain, so we need to enumerate all possible next characters
    # given the prefix_tokens (which are the tokens so far).
    # We want the probability distribution over the *next character*.

    # To do this, we need to enumerate all possible coin flips for the underlying DisplayChain,
    # and for each, generate the next character after the given prefix.

    # Get the underlying DisplayChain
    display_chain = chain.chain

    # Figure out the roots (stochastic coin flips) in the DisplayChain
    roots = [w for w in display_chain
             if isinstance(w, IndependentFeatureWrapper) and w.global_control is None]

    # Determine the position and previous token for the next token to generate
    prev_s   = prefix_tokens[-1] if prefix_tokens else ""
    position = len(prefix_tokens)

    # For each possible coin flip assignment, generate the next token, then
    # for each, get the next character (first char if buffer is empty, otherwise next char in buffer)
    from collections import defaultdict
    import itertools

    char_counts = defaultdict(int)

    if not roots:  # deterministic chain
        # Generate the next token deterministically
        token = display_chain("", s_prev=prev_s, position=position)
        # Now, simulate the CharStreamer buffer after prefix_tokens
        # We need to reconstruct the buffer: after prefix_tokens, the buffer is the next token
        # and we want the first character of that token
        if token:
            ch = token[0]
            char_counts[ch] = 1
        return {k: v for k, v in char_counts.items()}

    # Stochastic case: enumerate all coin flips
    for coin_vec in itertools.product([0, 1], repeat=len(roots)):
        # Freeze the chain for this coin flip assignment
        frozen_chain = _freeze_once(display_chain, coin_vec)
        # Generate the next token
        token = frozen_chain("", s_prev=prev_s, position=position)
        # The next character is the first character of the token
        if token:
            ch = token[0]
            char_counts[ch] += 1

    total = sum(char_counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in char_counts.items()}


def oracle_next_probs(chain: DisplayChain, prefix_tokens, *, seq_len=None):
    """
    Exact P(x_t | prefix) for *one* DisplayChain generation step.
    prefix_tokens : list[str]  (no blanks) – may be empty
    Returns dict{token: prob}.
    """
    prev_s   = prefix_tokens[-1] if prefix_tokens else ""
    position = len(prefix_tokens)

    # list all stochastic roots (per-step coin-flips)
    roots = [w for w in chain
             if isinstance(w, IndependentFeatureWrapper) and w.global_control is None]

    if not roots:                         # deterministic chain
        tok = chain("", s_prev=prev_s, position=position)
        return {tok: 1.0}

    probs = defaultdict(int)
    for coin_vec in itertools.product([0, 1], repeat=len(roots)):
        out = _freeze_once(chain, coin_vec)("", s_prev=prev_s, position=position)
        probs[out] += 1                   # each configuration equally likely

    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def oracle_entropy(probs):
    return -sum(p*math.log2(p) for p in probs.values())


def compare_batch(model, batch_tokens, char_streamer, tok2id, id2tok):
    # batch_tokens : (B,T)
    ent_oracle, kl_gap = [], []
    for b in range(batch_tokens.size(0)):
        prefix = [id2tok[i.item()] for i in batch_tokens[b,:-1]]
        probs  = oracle_next_probs_char(char_streamer, prefix)
        ent_oracle.append(oracle_entropy(probs))
        print(probs)
        with torch.no_grad():
            logits = model(batch_tokens[b,:-1].unsqueeze(0))[0,-1]  # (V,)
            logp   = logits.log_softmax(-1)
            gap = 0
            for tok, p in probs.items():
                gap += p * (math.log(p) - logp[tok2id[tok]].item())
            kl_gap.append(gap)
    return sum(ent_oracle)/len(ent_oracle), sum(kl_gap)/len(kl_gap)


SIGNS = {"+", "-", "?", "!", "<", ">", "[", "]"}

def chars_to_chain_tokens(char_list):
    tok, out = None, []
    for ch in char_list:
        if ch == " ":
            continue
        if ch in SIGNS:                # start new token with sign
            tok = ch
        else:                          # must be letter; finish token
            tok = (tok or "") + ch
            out.append(tok)
            tok = None
    return out