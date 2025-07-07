import einops
import torch
from transformer_lens import HookedTransformer
from transformer_lens import HookedTransformerConfig


def count_parameters(model: torch.nn.Module, requires_grad: bool = False) -> int:
    """Returns the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): PyTorch model to count parameters for
        requires_grad (bool): Whether to count only trainable parameters
    Returns:
        int: Total number of trainable parameters
    """
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def convert_nanogpt_weights(
    old_state_dict, cfg: HookedTransformerConfig, bias: bool = False
):
    """For https://github.com/karpathy/nanoGPT
    There are two complications with converting nanogpt models:
    The first is that some state dicts have an unwanted prefix on keys that needs to be removed.
    The second is that the models can be saved with or without bias. By default, there
    is no bias. This function can handle both cases."""
    # Nanogpt models saved after torch.compile() have this unwanted prefix
    # This is a simple way to remove it
    unwanted_prefix = "_orig_mod."
    for k, v in list(old_state_dict.items()):
        if k.startswith(unwanted_prefix):
            old_state_dict[k[len(unwanted_prefix) :]] = old_state_dict.pop(k)

    new_state_dict = {}
    new_state_dict["pos_embed.W_pos"] = old_state_dict["transformer.wpe.weight"]
    new_state_dict["embed.W_E"] = old_state_dict["transformer.wte.weight"]

    new_state_dict["ln_final.w"] = old_state_dict["transformer.ln_f.weight"]
    new_state_dict["ln_final.b"] = torch.zeros_like(
        old_state_dict["transformer.ln_f.weight"]
    )
    new_state_dict["unembed.W_U"] = old_state_dict["lm_head.weight"].T

    if bias:
        new_state_dict["ln_final.b"] = old_state_dict["transformer.ln_f.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"transformer.h.{layer}"

        new_state_dict[f"blocks.{layer}.ln1.w"] = old_state_dict[
            f"{layer_key}.ln_1.weight"
        ]
        # A bias of zeros is required for folding layer norm
        new_state_dict[f"blocks.{layer}.ln1.b"] = torch.zeros_like(
            old_state_dict[f"{layer_key}.ln_1.weight"]
        )
        new_state_dict[f"blocks.{layer}.ln2.w"] = old_state_dict[
            f"{layer_key}.ln_2.weight"
        ]
        new_state_dict[f"blocks.{layer}.ln2.b"] = torch.zeros_like(
            old_state_dict[f"{layer_key}.ln_2.weight"]
        )

        W = old_state_dict[f"{layer_key}.attn.c_attn.weight"]
        W_Q, W_K, W_V = torch.tensor_split(W.permute(1, 0), 3, dim=0)
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        new_state_dict[f"blocks.{layer}.attn.W_Q"] = W_Q
        new_state_dict[f"blocks.{layer}.attn.W_K"] = W_K
        new_state_dict[f"blocks.{layer}.attn.W_V"] = W_V

        W_O = old_state_dict[f"{layer_key}.attn.c_proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        new_state_dict[f"blocks.{layer}.attn.W_O"] = W_O

        new_state_dict[f"blocks.{layer}.mlp.W_in"] = old_state_dict[
            f"{layer_key}.mlp.c_fc.weight"
        ]
        new_state_dict[f"blocks.{layer}.mlp.W_out"] = old_state_dict[
            f"{layer_key}.mlp.c_proj.weight"
        ]

        # Add mask and IGNORE parameters with correct shape
        new_state_dict[f"blocks.{layer}.attn.mask"] = torch.tril(
            torch.ones(cfg.n_ctx, cfg.n_ctx)
        )
        new_state_dict[f"blocks.{layer}.attn.IGNORE"] = torch.tensor(0.0)

        if bias:
            new_state_dict[f"blocks.{layer}.ln1.b"] = old_state_dict[
                f"{layer_key}.ln_1.bias"
            ]
            new_state_dict[f"blocks.{layer}.ln2.b"] = old_state_dict[
                f"{layer_key}.ln_2.bias"
            ]
            new_state_dict[f"blocks.{layer}.mlp.b_in"] = old_state_dict[
                f"{layer_key}.mlp.c_fc.bias"
            ]
            new_state_dict[f"blocks.{layer}.mlp.b_out"] = old_state_dict[
                f"{layer_key}.mlp.c_proj.bias"
            ]

            B = old_state_dict[f"{layer_key}.attn.c_attn.bias"]
            B_Q, B_K, B_V = torch.tensor_split(B, 3, dim=0)
            B_Q = einops.rearrange(B_Q, "(i h)->i h", i=cfg.n_heads)
            B_K = einops.rearrange(B_K, "(i h)->i h", i=cfg.n_heads)
            B_V = einops.rearrange(B_V, "(i h)->i h", i=cfg.n_heads)
            new_state_dict[f"blocks.{layer}.attn.b_Q"] = B_Q
            new_state_dict[f"blocks.{layer}.attn.b_K"] = B_K
            new_state_dict[f"blocks.{layer}.attn.b_V"] = B_V
            new_state_dict[f"blocks.{layer}.attn.b_O"] = old_state_dict[
                f"{layer_key}.attn.c_proj.bias"
            ]

            # Add mask and IGNORE parameters with correct shape
            new_state_dict[f"blocks.{layer}.attn.mask"] = torch.ones(
                cfg.n_ctx, cfg.n_ctx
            )
            new_state_dict[f"blocks.{layer}.attn.IGNORE"] = torch.tensor(0.0)

    new_state_dict["unembed.b_U"] = torch.zeros(
        cfg.d_vocab
    )  # GPT-2 typically has no unembed bias

    return new_state_dict


def load_gpt2_to_hooked_transformer(hf_model, hook_config):
    """Load GPT2-style Hugging Face model into HookedTransformer."""
    hooked_model = HookedTransformer(hook_config)
    state_dict = hf_model.state_dict()
    new_state_dict = {}

    # Map embeddings
    new_state_dict["embed.W_E"] = state_dict["transformer.wte.weight"]
    new_state_dict["pos_embed.W_pos"] = state_dict["transformer.wpe.weight"]

    # Map transformer block parameters
    for i in range(hook_config.n_layers):
        new_state_dict[f"blocks.{i}.ln1.w"] = state_dict[
            f"transformer.h.{i}.ln_1.weight"
        ]
        new_state_dict[f"blocks.{i}.ln1.b"] = state_dict[f"transformer.h.{i}.ln_1.bias"]
        new_state_dict[f"blocks.{i}.ln2.w"] = state_dict[
            f"transformer.h.{i}.ln_2.weight"
        ]
        new_state_dict[f"blocks.{i}.ln2.b"] = state_dict[f"transformer.h.{i}.ln_2.bias"]

        # Attention weights
        attn_weights = state_dict[f"transformer.h.{i}.attn.c_attn.weight"]
        attn_bias = state_dict[f"transformer.h.{i}.attn.c_attn.bias"]
        d_model = hook_config.d_model
        d_head = hook_config.d_head
        n_heads = hook_config.n_heads

        Wq, Wk, Wv = attn_weights.split(d_model, dim=1)
        # [d, d] -> [d, 3*d] -> [d,d], [d,d], [d,d] -> 3x[d, d_head, d_model]
        # vs split in transformer_lens
        # 3x([d, d] -> [d, d] -> [d, d_head, d_model])
        new_state_dict[f"blocks.{i}.attn.W_Q"] = Wq.view(
            d_model, n_heads, d_head
        ).permute(1, 0, 2)
        #     n_heads,
        #     d_head,
        #     d_model,
        # ).permute(0, 2, 1)
        # n_heads x d_model x d_head
        new_state_dict[f"blocks.{i}.attn.W_K"] = Wk.view(
            d_model, n_heads, d_head
        ).permute(1, 0, 2)

        new_state_dict[f"blocks.{i}.attn.W_V"] = Wv.view(
            d_model, n_heads, d_head
        ).permute(1, 0, 2)

        new_state_dict[f"blocks.{i}.attn.b_Q"] = attn_bias[:d_model].view(
            n_heads, d_head
        )
        new_state_dict[f"blocks.{i}.attn.b_K"] = attn_bias[d_model : 2 * d_model].view(
            n_heads, d_head
        )
        new_state_dict[f"blocks.{i}.attn.b_V"] = attn_bias[2 * d_model :].view(
            n_heads, d_head
        )

        # Attention output weights
        new_state_dict[f"blocks.{i}.attn.W_O"] = (
            state_dict[f"transformer.h.{i}.attn.c_proj.weight"].view(
                n_heads, d_head, d_model
            )
            #     d_model,
            #     n_heads,
            #     d_head,
            # )
            # .permute(1, 2, 0)
        )
        new_state_dict[f"blocks.{i}.attn.b_O"] = state_dict[
            f"transformer.h.{i}.attn.c_proj.bias"
        ]

        # Add mask and IGNORE parameters with correct shape
        new_state_dict[f"blocks.{i}.attn.mask"] = torch.tril(
            torch.ones(hook_config.n_ctx, hook_config.n_ctx)
        )
        new_state_dict[f"blocks.{i}.attn.IGNORE"] = torch.tensor(0.0)

        # MLP weights
        new_state_dict[f"blocks.{i}.mlp.W_in"] = state_dict[
            f"transformer.h.{i}.mlp.c_fc.weight"
        ]
        new_state_dict[f"blocks.{i}.mlp.b_in"] = state_dict[
            f"transformer.h.{i}.mlp.c_fc.bias"
        ]
        new_state_dict[f"blocks.{i}.mlp.W_out"] = state_dict[
            f"transformer.h.{i}.mlp.c_proj.weight"
        ]
        new_state_dict[f"blocks.{i}.mlp.b_out"] = state_dict[
            f"transformer.h.{i}.mlp.c_proj.bias"
        ]

    # Map final layer norm
    new_state_dict["ln_final.w"] = state_dict["transformer.ln_f.weight"]
    new_state_dict["ln_final.b"] = state_dict["transformer.ln_f.bias"]

    # Map unembedding weights
    new_state_dict["unembed.W_U"] = state_dict["lm_head.weight"].T
    new_state_dict["unembed.b_U"] = torch.zeros(
        hook_config.d_vocab
    )  # GPT-2 typically has no unembed bias

    # Load the mapped state dict into HookedTransformer
    hooked_model.load_state_dict(new_state_dict)
    return hooked_model
