import math
import torch
import numpy as np
import torch.nn as nn
import numpy.typing as npt
from einops import einsum, rearrange

class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(
            torch.empty(self.out_features, self.in_features, device=device, dtype=dtype),
            requires_grad=True
        )

        self.reset_params()

    def reset_params(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(
            self.W,
            mean=0.0, 
            std=std, 
            a=-3.0 * std, 
            b=3.0 * std
        )

    def forward(self, x: torch.Tensor):
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None, requires_grad=True):
        super(EmbeddingLayer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.E = nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype),
            requires_grad=requires_grad
        )

        self.reset_params()

    def reset_params(self):
        nn.init.trunc_normal_(
            self.E,
            mean=0.0,
            std=1.0,
            a=-3,
            b=3
        )

    def forward(self, x: torch.Tensor):
        return self.E[x]

class RMSNormLayer(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNormLayer, self).__init__()

        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype),
            requires_grad=True
        )

        self.reset_params()

    def reset_params(self):
        nn.init.ones_(self.g)

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x_float = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps)
        rmsnorm = (x_float / rms) * self.g.to(torch.float32)
        return rmsnorm.to(in_dtype)

def SiLU(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = LinearLayer(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.W2 = LinearLayer(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.W3 = LinearLayer(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def SiLU(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        silg_x_W1 = self.SiLU(self.W1(x))
        x_W3 = self.W3(x)
        element_wise_mult = torch.mul(silg_x_W1, x_W3)
        return self.W2(element_wise_mult)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RoPE, self).__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        thetas = einsum(t, inv_freq, "i,j -> i j")

        self.register_buffer("cos_buffer", torch.cos(thetas), persistent=False)
        self.register_buffer("sin_buffer", torch.sin(thetas), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        cos_sliced = self.cos_buffer[token_positions].to(dtype=x.dtype)
        sin_sliced = self.sin_buffer[token_positions].to(dtype=x.dtype)

        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]

        out0 = x0 * cos_sliced - x1 * sin_sliced
        out1 = x0 * sin_sliced + x1 * cos_sliced

        out_combined = torch.stack([out0, out1], dim=-1)
        return out_combined.reshape(x.shape)

def SoftmaxLayer(in_features: torch.Tensor, dim: int):
    max_val_in_dim = torch.max(in_features, dim=dim, keepdim=True).values
    adjusted_values = in_features - max_val_in_dim
    exp_values = torch.exp(adjusted_values)
    sum_values = torch.sum(exp_values, dim=dim, keepdim=True)
    return exp_values / sum_values

def AttentionLayer(Query: torch.Tensor, Key: torch.Tensor, Value: torch.Tensor, mask=None, dim=None):
    d_k = Query.shape[-1]
    Q_K_T = einsum(Query, Key, " ... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    if mask is not None: masked_Q_K_T = torch.where(mask, Q_K_T, float('-inf'))
    else: masked_Q_K_T = Q_K_T
    probs = SoftmaxLayer(masked_Q_K_T, dim=-1)
    return einsum(probs, Value, " ... n m, ... m d_v -> ... n d_v")

class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = None, device=None, dtype=None):
        super(MultiHeadSelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = self.d_model // self.h
        self.d_v = self.d_model // self.h
        self.max_seq_len = max_seq_len
        self.device = device

        self.Query = LinearLayer(in_features=self.d_model, out_features=self.d_k * self.h, device=device, dtype=dtype)
        self.Key = LinearLayer(in_features=self.d_model, out_features=self.d_k * self.h, device=device, dtype=dtype)
        self.Value = LinearLayer(in_features=self.d_model, out_features=self.d_v * self.h, device=device, dtype=dtype)

        self.Output = LinearLayer(in_features=self.d_v * self.h, out_features=self.d_model, device=device, dtype=dtype)

        if theta is not None: self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=self.max_seq_len, device=device)

    def forward(self, x, token_positions=None):
        Q_out = self.Query(x)
        K_out = self.Key(x)
        V_out = self.Value(x)
        Q_out = rearrange(Q_out, " ... seq (heads d_k) -> ... heads seq d_k", heads=self.h)
        K_out = rearrange(K_out, " ... seq (heads d_k) -> ... heads seq d_k", heads=self.h)
        V_out = rearrange(V_out, " ... seq (heads d_v) -> ... heads seq d_v", heads=self.h)

        seq_len = x.shape[-2]
        if token_positions is not None:
            Q_out = self.rope(Q_out, token_positions)
            K_out = self.rope(K_out, token_positions)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))
        multihead_attention = AttentionLayer(Q_out, K_out, V_out, mask=mask)
        multihead_attention = rearrange(multihead_attention, " ... heads seq d_v -> ... seq (heads d_v)", heads=self.h)
        return self.Output(multihead_attention)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # layer 1
        self.rmsnorm1 = RMSNormLayer(d_model=self.d_model, device=device, dtype=dtype)
        self.multihead = MultiHeadSelfAttentionLayer(
                            d_model=self.d_model,
                            num_heads=self.num_heads,
                            max_seq_len=self.max_seq_len,
                            theta=self.theta,
                            device=device,
                            dtype=dtype
                        )
        ## layer 2
        self.rmsnorm2 = RMSNormLayer(d_model=self.d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        
        seq_len = x.shape[-2]
        if token_positions is None: token_positions = torch.arange(seq_len, device=x.device)
        
        y = x + self.multihead(self.rmsnorm1(x), token_positions)
        out = y + self.ffn(self.rmsnorm2(y))
        return out

class TransformerLM(nn.Module):
    def __init__(
            self, 
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device=None,
            dtype=None
        ):
        super(TransformerLM, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embedding = EmbeddingLayer(num_embeddings=self.vocab_size, embedding_dim=self.d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                max_seq_len=self.context_length,
                theta=self.rope_theta,
                device=device,
                dtype=dtype
            ) for _ in range(self.num_layers)
        ])
        self.norm = RMSNormLayer(d_model=self.d_model, device=device, dtype=dtype)
        self.linear = LinearLayer(in_features=self.d_model, out_features=self.vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.linear(x)
        return x

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0: raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)                
        super(AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            alpha = group['lr']
            eps = group['eps']
            lambd = group['weight_decay']

            for p in group['params']:

                state = self.state[p]
                g = p.grad
                
                if g is None: continue

                if len(state) == 0:
                    state['t'] = 1
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                t, m, v = state['t'], state['m'], state['v']

                # compute the moments
                m.mul_(beta1).add_(g, alpha = 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value = 1 - beta2)

                # update the learning rate
                correction_numerator = 1 - beta2 ** t
                correction_denominator = 1 - beta1 ** t
                alpha_updated = alpha * (correction_numerator ** 0.5) / correction_denominator

                # update parameters and apply weight decay
                p.addcdiv_(m, v.sqrt().add_(eps), value = -alpha_updated)
                if lambd != 0: p.add_(p, alpha = -alpha * lambd)
                state['t'] = t + 1
                
        return loss

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    adjusted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum = torch.log(torch.exp(adjusted).sum(dim=-1))
    target_logits = adjusted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum - target_logits).mean()

def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)
    else: return min_learning_rate


def gradient_clipping(parameters, max_l2_norm: float, eps: float = 1e-6) -> None:
    
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0: return

    total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):

    n = len(dataset)
    starting_idxs = np.random.randint(0, n - context_length, batch_size)

    inputs = np.stack([dataset[s : s + context_length] for s in starting_idxs])
    targets = np.stack([dataset[s + 1 : s + context_length + 1] for s in starting_idxs])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return (inputs, targets)