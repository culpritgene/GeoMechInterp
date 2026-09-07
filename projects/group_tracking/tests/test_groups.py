"""Tests for the group-tracking package.  Plain functions; run with

    .venv/bin/python projects/group_tracking/tests/test_groups.py

(pytest also collects them if installed)."""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from groups import cyclic, dihedral, make_group, torus, verify  # noqa: E402
from group_data import (BOS, EOS, PAD, SEP, GroupData, answer_positions, decode, encode, generate,  # noqa: E402
                        loss_mask, sample_minimal_pairs, sample_words, write_dataset)

GROUPS = ["Z12", "Z36", "Z360", "T36x12", "D36", "D7"]


def _brute_assoc(G, n_samples=20000, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = (rng.integers(0, G.order, n_samples) for _ in range(3))
    return (G.table[G.table[a, b], c] == G.table[a, G.table[b, c]]).all()


def test_table_closure_and_associativity():
    for name in GROUPS:
        G = make_group(name)
        assert verify(G)["associative"]
        assert _brute_assoc(G)
        n = G.order
        assert (np.sort(G.table, 1) == np.arange(n)).all() and (np.sort(G.table, 0) == np.arange(n)[:, None]).all()
    assert make_group("D36").is_abelian is False and make_group("Z36").is_abelian and make_group("T36x12").is_abelian


def test_dihedral_relations():
    G = dihedral(36)
    r, s = 1, 36
    assert G.product([r] * 36) == G.identity and G.product([s, s]) == G.identity
    # s r s = r^{-1}
    assert G.product([s, r, s]) == G.inverse[r]
    # sr generator is s . r
    sr = G.generators[G.generator_names.index("sr")]
    assert sr == G.table[s, r]
    assert not G.commute(r, s) and G.commute(r, 2)
    # reflections have eps = -1
    eps = G.sign
    assert (eps[36:] == -1).all() and (eps[:36] == 1).all()


def test_representation_homomorphism():
    for name in GROUPS:
        G = make_group(name)
        T = G.table
        for irr, R in G.irreps.items():
            prod = np.einsum("gij,hjk->ghik", R, R)
            assert np.allclose(prod, R[T], atol=1e-9), (name, irr)
            assert np.allclose(R[G.identity], np.eye(R.shape[1]))
            # irreps are faithful enough to separate: rho_1 distinct on all elements for Z_n and D_n
        if name.startswith(("Z", "D")):
            R1 = G.irreps["rho1"].reshape(G.order, -1)
            assert len(np.unique(np.round(R1, 9), axis=0)) == G.order
    # D_n coordinates are exactly the irrep matrix entries (c, -eps s, s, eps c) reordered
    G = dihedral(36)
    R = G.irreps["rho2"]
    c, s, ec, es = (G.coords[:, G.coord_groups["f2"][i]] for i in range(4))
    assert np.allclose(R[:, 0, 0], c) and np.allclose(R[:, 1, 0], s) and np.allclose(R[:, 0, 1], -es) and np.allclose(R[:, 1, 1], ec)
    # sign character equals the determinant of every 2-D irrep
    assert np.allclose(np.linalg.det(R), G.sign)


def test_inverse_consistency():
    for name in GROUPS:
        G = make_group(name)
        ar = np.arange(G.order)
        assert (G.table[ar, G.inverse] == G.identity).all() and (G.table[G.inverse, ar] == G.identity).all()
        assert (G.inverse[G.inverse] == ar).all()
        # (gh)^-1 = h^-1 g^-1
        assert (G.inverse[G.table] == G.table[G.inverse[None, :], G.inverse[:, None]]).all()
        # generator set closed under inversion
        assert set(G.inverse[G.generators].tolist()) == set(G.generators.tolist())


def test_sympy_cross_check():
    """Optional: compare D_36 and Z_36 tables with sympy's permutation groups."""
    try:
        from sympy.combinatorics import Permutation
    except ImportError:
        print("  (sympy missing, skipped)"); return
    n = 36
    r = Permutation([(i + 1) % n for i in range(n)])
    s = Permutation([(-i) % n for i in range(n)])
    G = dihedral(n)
    perms = [(r ** int(a)) * (s ** int(b)) for g in range(G.order) for a, b in [(g % n, g // n)]]
    assert len({p for p in perms}) == G.order
    rng = np.random.default_rng(0)
    for g, h in rng.integers(0, G.order, (400, 2)):
        assert perms[g] * perms[h] == perms[G.table[g, h]]
    Z = cyclic(n)
    zp = [r ** int(a) for a in range(n)]
    for g, h in rng.integers(0, n, (200, 2)):
        assert zp[g] * zp[h] == zp[Z.table[g, h]]


def test_generators_and_probs():
    G = dihedral(36, p_reflect=0.2)
    p = dict(zip(G.generator_names, G.generator_probs))
    assert abs(p["s"] + p["sr"] - 0.2) < 1e-12 and abs(sum(p.values()) - 1) < 1e-12
    Z = cyclic(36)
    assert Z.generators.tolist() == [1, 35, 2, 34, 3, 33]
    T = torus(36, 12, diagonals=True)
    assert T.n_gen == 8 and len(set(T.generators.tolist())) == 8
    T = torus(36, 12)
    assert T.n_gen == 4


def test_data_prefix_products():
    G = make_group("D36")
    arrays = generate(G, seed=1, n_train=300, n_eval=100, kmax=12)
    tok, k, prefix = arrays["tokens"].astype(int), arrays["k"].astype(int), arrays["prefix"].astype(int)
    for i in range(len(tok)):
        e0, word, ek = decode(G, tok[i])
        pp = G.prefix_products(e0, word)
        assert len(word) == k[i]
        assert (pp == prefix[i, :k[i] + 1]).all() and (prefix[i, k[i] + 1:] == -1).all()
        assert ek == pp[-1]
        if e0 == G.identity:
            assert ek == G.product([G.generators[j] for j in word])
        assert np.allclose(arrays["prefix_irrep"][i, :k[i] + 1], G.coords[pp]) and (arrays["prefix_irrep"][i, k[i] + 1:] == 0).all()
    # held-out bigrams: absent from train/val/test, present in every held-out row
    pairs = {tuple(p) for p in arrays["heldout_bigrams"].tolist()}
    split = arrays["split"]
    for i in range(len(tok)):
        _, word, _ = decode(G, tok[i])
        big = set(zip(word[:-1], word[1:]))
        if split[i] in ("train", "validation", "test", "minimal_pair"):
            assert not (big & pairs), (split[i], word)
        elif split[i] == "heldout_bigram":
            assert big & pairs
    # minimal pairs: same e0, same length, adjacent swap of non-commuting generators, different answers
    pid = arrays["pair_id"]
    mp = np.where(split == "minimal_pair")[0]
    assert len(mp) == 100 and (pid[mp[0::2]] == pid[mp[1::2]]).all()
    for i, j in zip(mp[0::2], mp[1::2]):
        e0a, wa, eka = decode(G, tok[i]); e0b, wb, ekb = decode(G, tok[j])
        assert e0a == e0b and len(wa) == len(wb) and eka != ekb
        diff = [t for t in range(len(wa)) if wa[t] != wb[t]]
        assert len(diff) == 2 and diff[1] == diff[0] + 1 and wa[diff[0]] == wb[diff[1]] and wa[diff[1]] == wb[diff[0]]
        ga, gb = G.generators[wa[diff[0]]], G.generators[wa[diff[1]]]
        assert not G.commute(ga, gb)
    # abelian groups have no minimal pairs
    Z = make_group("Z36")
    k, w, _ = sample_minimal_pairs(np.random.default_rng(0), Z, 10, 12, np.zeros((6, 6), bool))
    assert len(k) == 0
    # word sampling honours the forbidden bigram matrix and length range
    forb = np.zeros((8, 8), bool); forb[0, 1] = True
    k, w = sample_words(np.random.default_rng(0), G, 2000, 12, forb)
    assert k.min() >= 1 and k.max() <= 12 and (w[:, 0] >= 0).all()
    assert not any((w[i, t] == 0 and w[i, t + 1] == 1) for i in range(2000) for t in range(k[i] - 1))


def test_mask_correctness():
    G = make_group("Z36")
    rows = [encode(G, 5, [0, 1, 2], 12), encode(G, 0, [3], 12), encode(G, 35, list(range(6)) * 2, 12)]
    X = torch.from_numpy(np.stack(rows))
    m = loss_mask(X)
    assert m.shape == (3, X.shape[1] - 1) and (m.sum(1) == 1).all()
    pos = answer_positions(X)
    for i, (e0, word) in enumerate([(5, [0, 1, 2]), (0, [3]), (35, list(range(6)) * 2)]):
        k = len(word)
        assert int(pos[i]) == 3 + k and X[i, 3 + k] == SEP and X[i, 4 + k] == 4 + G.prefix_products(e0, word)[-1]
        tgt = X[i, 1:][m[i]]
        assert tgt.tolist() == [4 + G.prefix_products(e0, word)[-1]]
        assert X[i, 0] == BOS and X[i, 5 + k] == EOS and (X[i, 6 + k:] == PAD).all()
    assert (answer_positions(X.numpy()) == pos.numpy()).all()


def test_npz_round_trip():
    G = make_group("T36x12")
    arrays = generate(G, seed=3, n_train=200, n_eval=50, kmax=8)
    with tempfile.TemporaryDirectory(dir="/var/tmp") as d:
        p = Path(d) / "t.npz"
        write_dataset(p, arrays)
        gd = GroupData(p)
        assert gd.group.name == "T36x12" and gd.kmax == 8 and gd.vocab_size == 4 + 432 + 4 and gd.block_size == 14
        assert (gd.group.table == G.table).all() and gd.summary()["train"] == 200
        for split in ("train", "validation", "test", "heldout_bigram"):
            X, idx = gd.tensors(split, 20)
            for r, i in enumerate(idx):
                e0, word, ek = gd.decode(i)
                assert (X[r].numpy() == encode(G, e0, word, 8)).all()
                assert ek == gd.final_elements([i])[0] == G.prefix_products(e0, word)[-1]
                assert gd.generator_positions(i).tolist() == list(range(3, 3 + len(word))) and gd.final_sep_position(i) == 3 + len(word)
        assert gd.summary()["minimal_pair"] == 0     # abelian
    assert decode(G, [BOS, 4, SEP, SEP, 4, EOS]) is None and decode(G, [BOS, 4, SEP, 4 + 432, SEP, 4 + 12, EOS]) == (0, [0], 12)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
