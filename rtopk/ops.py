
import torch
__all__ = ["rtopk"]

def rtopk(data, k, max_iter=10, precision=1e-5):
    data_in = data.flatten(0, -2)
    values, indices = torch.ops.rtopk.rtopk.default(data_in, k, max_iter, precision)
    return values.unflatten(0, data.shape[:-1]), indices.unflatten(0, data.shape[:-1])


@torch.library.register_fake("rtopk::rtopk")
def _(data, k, max_iter=10, precision=1e-5):
    torch._check(0 <= k <= data.shape[-1])
    torch._check(data.dtype in (torch.float32, torch.bfloat16))
    values = torch.empty(
        data.shape[:-1] + (k,),
        dtype=data.dtype,
        device=data.device,
    )
    indices = torch.empty(
        data.shape[:-1] + (k,),
        dtype=torch.int32,
        device=data.device,
    )
    return values, indices


def _backward_rtopk(ctx, grad_output, something):
    if not ctx.needs_input_grad[0]:
        return None, None, None, None
    indices, = ctx.saved_tensors  # indices has shape [N, k]
    # Initialize gradient for the input as zeros.
    grad_input = torch.zeros(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
    # Scatter the gradient from the top-k positions back to the corresponding indices.
    # This is a straight-through estimator.
    grad_input.scatter_(1, indices, grad_output)
    # No gradients are needed for k, max_iter, or precision.
    return grad_input, None, None, None


def _setup_context(ctx, inputs, output):
    if ctx.needs_input_grad[0]:
        ctx.save_for_backward(output[1])


torch.library.register_autograd(
    "rtopk::rtopk",
    _backward_rtopk,
    setup_context=_setup_context,
)
