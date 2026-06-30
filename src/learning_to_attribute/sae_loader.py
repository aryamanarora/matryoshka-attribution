"""Load Llama Scope per-layer residual JumpReLU SAEs (fnlp/Llama3_1-8B-Base-LXR-8x)."""
import json, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class LlamaScopeSAE:
    def __init__(self, repo, layer, device, dtype=torch.bfloat16, component="R"):
        # component: "R" residual stream (LXR repo), "M" MLP output (LXM repo)
        sub = f"Llama3_1-8B-Base-L{layer}{component}-8x"
        sd = load_file(hf_hub_download(repo, f"{sub}/checkpoints/final.safetensors"))
        hp = json.load(open(hf_hub_download(repo, f"{sub}/hyperparams.json")))
        self.W_enc = sd["encoder.weight"].to(device, dtype)   # [d_sae, d_model]
        self.b_enc = sd["encoder.bias"].to(device, dtype)
        self.W_dec = sd["decoder.weight"].to(device, dtype)   # [d_model, d_sae]
        self.b_dec = sd["decoder.bias"].to(device, dtype)
        self.thresh = hp["jump_relu_threshold"]
        self.s = (self.W_enc.shape[1] ** 0.5) / hp["dataset_average_activation_norm"]["in"]
        self.d_sae = self.W_enc.shape[0]

    def encode(self, x):                                      # x: [..., d_model]
        pre = torch.relu((x * self.s) @ self.W_enc.T + self.b_enc)
        return pre * (pre > self.thresh)                      # [..., d_sae]

    def decode(self, f):
        return (f @ self.W_dec.T + self.b_dec) / self.s

    def decode_delta(self, g):                                # affine part only (b_dec cancels)
        return (g.to(self.W_dec.dtype) @ self.W_dec.T) / self.s


def load_llama_scope_saes(repo, n_layers, device, dtype=torch.float32, component="R"):
    # float32 by default: the bf16 interchange overflowed to NaN when feature
    # deltas are injected across all 32 layers during training.
    return {li: LlamaScopeSAE(repo, li, device, dtype, component) for li in range(n_layers)}
