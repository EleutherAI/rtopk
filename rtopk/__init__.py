import torch
import rtopk_cuda

class RTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, k, max_iter=10, precision=1e-5):
        # Call the CUDA extension that returns both values and indices.
        # data: [N, dim_origin]
        values, indices = rtopk_cuda.rtopk_forward(data, k, max_iter, precision)
        # Save the indices and input shape for backward.
        ctx.save_for_backward(indices)
        ctx.input_shape = data.shape
        return values, indices

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors  # indices has shape [N, k]
        # Initialize gradient for the input as zeros.
        grad_input = torch.zeros(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
        # Scatter the gradient from the top-k positions back to the corresponding indices.
        # This is a straight-through estimator.
        grad_input.scatter_(1, indices, grad_output)
        # No gradients are needed for k, max_iter, or precision.
        return grad_input, None, None, None

def rtopk_autograd(data, k, max_iter=10, precision=1e-5):
    higher_dimensions = data.dim() > 2
    if higher_dimensions:
        og_shape = data.shape[:-1]
        data = data.flatten(0, -2)
    values, indices = RTopKFunction.apply(data, k, max_iter, precision)
    if higher_dimensions:
        values = values.unflatten(0, og_shape)
        indices = indices.unflatten(0, og_shape)
    return values, indices


__all__ = ["rtopk_autograd", "RTopKFunction", "rtopk_cuda"]