#%%
%env CUDA_VISIBLE_DEVICES=2
%env CUDA_LAUNCH_BLOCKING=1
%load_ext autoreload
import torch
import rtopk
%autoreload 2
#%%
data = torch.randn((128, 256), device="cuda", requires_grad=True)
torch.allclose(rtopk.ops.rtopk(    data, 16)[0].min(dim=-1).values,
               data.topk(16).values.min(dim=-1).values,
               atol=1e-2)
#%%
B = 2**14
# E = 65536 // 2
E = 4096
K = 128

data_bf16 = torch.randn(B, E, device="cuda", requires_grad=True, dtype=torch.bfloat16)
data = data_bf16.float()
#%%
from tqdm.auto import tqdm
import torch.utils.benchmark


def timing(name, fn, *args):
    with torch.inference_mode():
        fn(*args)
        timer = torch.utils.benchmark.Timer(
            stmt="fn(*args)",
            globals={"fn": fn, "args": args},
            setup="fn(*args)",
        ).blocked_autorange()
        print(name, timer.mean * 1e3, "ms")
    return timer.mean


@torch.compile
def basic_topk(data):
    return data.topk(K)

@torch.compile
def groupmax(vals, k=K):
    M = vals.shape[-1]
    grouped = vals.unflatten(-1, (k, M // k))
    values, indices = grouped.max(dim=-1)
    indices = indices + torch.arange(0, k, device=indices.device, dtype=indices.dtype) * (M // k)
    return values, indices

MAX_SIZE = 8192
@torch.compile
def rtopk_topk(data, max_iter=10, k_div: int = 1, k: int = K):
    if data.shape[-1] < MAX_SIZE:
        return rtopk.ops.rtopk(data.float(), k, max_iter=max_iter)
    # return rtopk.rtopk_autograd(data.float(), k, max_iter=max_iter)
    # data = data.float()
    data = data.unflatten(-1, (-1, MAX_SIZE))
    # if k == k_div:
    #     values, indices = data.max(dim=-1)
    #     values, indices = values.unsqueeze(-1), indices.unsqueeze(-1)
    # else:
    values, indices = rtopk.ops.rtopk(data, k // k_div, max_iter=max_iter)
    # values, indices = data.topk(K // k_div)
    
    # return values, indices
    # return values[:, 0], indices[:, 0]
    # indices = indices.long()
    values_l2, indices_l2 = rtopk.rtopk_autograd(values.flatten(-2), k, max_iter=max_iter)
    # values_l2, indices_l2 = values.flatten(-2).topk(k)
    indices = (
        indices
        + torch.arange(data.shape[-2],
                       device=indices.device, dtype=torch.int32
                       )[:, None] * data.shape[-1]
        ).flatten(-2).gather(-1, indices_l2.long())
    return values_l2, indices

def recall(a_ind, b_ind):
    return (a_ind[..., None] == b_ind[..., None, :]).any(dim=-1).float().mean()

max_iters = [1, 2, 3, 4, 6, 8, 12, 16, 512, 2048]
rtopk_perfs = []
rtopk_recalls = []
k_div = 1
topk_perf = timing("topk", basic_topk, data_bf16)
groupmax_perf = timing("groupmax", groupmax, data_bf16)
with torch.inference_mode():
    vb, ib = basic_topk(data)
    vg, ig = groupmax(data)
    groupmax_recall = recall(ig, ib).item()
for max_iter in tqdm(max_iters):
    rtopk_perf = timing("rtopk", rtopk_topk, data_bf16, max_iter, k_div)
    with torch.inference_mode():
        vr, ir = rtopk_topk(data_bf16, max_iter=max_iter, k_div=k_div)
        rtopk_perfs.append(rtopk_perf)
        rtopk_recalls.append(recall(ir, ib).item())
    if rtopk_perf > topk_perf * 2:
        break
from matplotlib import pyplot as plt
plt.scatter(rtopk_perfs, rtopk_recalls)
plt.scatter(groupmax_perf, groupmax_recall, marker="x")
plt.plot([topk_perf, topk_perf], [0, 1], "--")
plt.ylabel("Recall")
plt.xlabel("Time (s)")
plt.xscale("log")
plt.title(f"RTopK Recall vs Time (B={B}, M={E}, K={K}, k_div={k_div})")
#%%
