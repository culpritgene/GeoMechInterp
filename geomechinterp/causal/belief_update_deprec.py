
def freeze_roots(chain: DisplayChain, root_assign: dict[str,int]) -> DisplayChain:
    frozen_funcs = []
    for w in chain:
        if (isinstance(w, IndependentFeatureWrapper)
                and w.global_control is None):            # a stochastic root
            name   = w.feature_fn.__name__
            value  = root_assign[name]                    # 0 or 1
            frozen_funcs.append(IndependentFeatureWrapper(w.feature_fn, value))
        else:
            frozen_funcs.append(w)
    return DisplayChain(frozen_funcs)


def eval_chain(chain, root_assign, seq_len=20):
    """
    Generate a full sequence deterministically given a root_assign dict.
    """
    if root_assign:                                       # make a frozen copy
        chain = freeze_roots(chain, root_assign)

    seq, prev = [], ""
    for pos in range(seq_len):
        token = chain("", s_prev=prev, position=pos)      # ❶ use the chain!
        seq.append(token)
        prev = token
    return seq


def oracle_next_probs(chain, prefix, seq_len=20):
    """
    Return dict{token: prob} for x_t given observed prefix.
    Works even when the chain has zero IndependentFeatureWrappers.
    """
    if seq_len < len(prefix) + 1:
        seq_len = len(prefix) + 1
    # 1. identify stochastic roots
    root_feats = [w.feature_fn.__name__
                  for w in chain
                  if isinstance(w, IndependentFeatureWrapper)
                    and w.global_control is None]

    # ---------- deterministic chain: no roots ------------------------------
    if not root_feats:
        seq = eval_chain(chain, root_assign={}, seq_len=seq_len)
        # Relaxed: allow prefix to be a partial match, as long as it matches up to the available sequence
        min_len = min(len(prefix), len(seq))
        if seq[:min_len] != prefix[:min_len]:
            warnings.warn("Prefix incompatible with deterministic chain")
            return {}
        
        # If prefix is longer than seq, cannot continue
        if len(prefix) >= len(seq):
            warnings.warn("Prefix longer than deterministic sequence")
            return {}
        return {seq[len(prefix)]: 1.0}          # point-mass

    # ---------- stochastic chain: enumerate roots --------------------------
    probs, mass = {}, 0
    for assignment in itertools.product([0, 1], repeat=len(root_feats)):
        root_assign = dict(zip(root_feats, assignment))
        seq = eval_chain(chain, root_assign, seq_len)
        if seq[:len(prefix)] == prefix:
            nxt = seq[len(prefix)]
            probs[nxt] = probs.get(nxt, 0) + 1
            mass += 1

    # normalise
    for k in probs:
        probs[k] /= mass
    return probs
