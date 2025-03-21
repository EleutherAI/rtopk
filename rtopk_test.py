#%%
%env CUDA_VISIBLE_DEVICES=5
%env CUDA_LAUNCH_BLOCKING=1
%load_ext autoreload
%autoreload 2
#%%
import torch
import rtopk  # This is the module built from your CUDA extension

data = torch.randn((128, 256), device="cuda", requires_grad=True)
torch.allclose(rtopk.rtopk_autograd(data, 16)[0].min(dim=-1).values, data.topk(16).values.min(dim=-1).values, atol=1e-2)
#%%
B = 2**12
E = 65536
K = 128

data_bf16 = torch.randn(B, E, device="cuda", requires_grad=True, dtype=torch.bfloat16)
data = data_bf16.float()
#%%
import torch.utils.benchmark

def timing(name, fn, *args):
    fn(*args)
    timer = torch.utils.benchmark.Timer(
        stmt="fn(*args)",
        globals={"fn": fn, "args": args},
        setup="fn(*args)",
    ).blocked_autorange()
    print(name, timer.mean * 1e3, "ms")


@torch.compile
def basic_topk(data):
    return data.topk(K)

@torch.compile
def groupmax(vals, k=K):
    M = vals.shape[-1]
    grouped = vals.view(-1, k, M // k)
    values, indices = grouped.max(dim=-1)
    indices = indices + torch.arange(0, k, device=indices.device, dtype=indices.dtype) * (M // k)
    return values.flatten(-2), indices.flatten(-2)

MAX_SIZE = 8192
@torch.compile
def rtopk_topk(data, k=K, max_iter=10):
    # return rtopk.rtopk_autograd(data.float(), k, max_iter=max_iter)
    data = data.float()
    data = data.unflatten(-1, (-1, MAX_SIZE))
    values, indices = rtopk.rtopk_autograd(data, k, max_iter=max_iter)
    # return values, indices
    # return values[:, 0], indices[:, 0]
    indices = indices.long()
    # values_l2, indices_l2 = rtopk.rtopk_autograd(values.flatten(-2), k, max_iter=max_iter)
    values_l2, indices_l2 = values.flatten(-2).topk(k)
    indices = (
        indices
        + torch.arange(indices.shape[-2],
                       device=indices.device, dtype=indices.dtype
                       )[:, None] * indices.shape[-1]
        ).flatten(-2).gather(-1, indices_l2.long())
    return values_l2, indices

with torch.inference_mode():
    timing("rtopk", rtopk_topk, data)
    timing("groupmax", groupmax, data_bf16)
    timing("topk", basic_topk, data_bf16)
    print(
        rtopk_topk(data)[0].min(dim=-1).values
        - basic_topk(data)[0].min(dim=-1).values
    )
    v = rtopk_topk(data)
v
# %%
v[1].max()