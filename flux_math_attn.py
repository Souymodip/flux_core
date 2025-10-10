import torch
from einops import rearrange
from torch import Tensor
import torch.nn.functional as F


# def create_grouped_attention_mask(seq_len: int, group_size: int = 64, 
#                                    free_groups: int = 2, 
#                                    device: torch.device = None, 
#                                    dtype: torch.dtype = None) -> Tensor:
#     """
#     Create an efficient attention mask for grouped attention pattern.
    
#     Args:
#         seq_len: Total sequence length
#         group_size: Size of each group (64)
#         free_groups: Number of initial groups with unrestricted attention (2)
#         device: Device to create mask on
#         dtype: Data type of mask (bool is most efficient for Flash Attention)
    
#     Returns:
#         Boolean mask of shape (seq_len, seq_len) where True = attend, False = mask out
#     """
#     # Create mask as bool for memory efficiency
#     mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    
#     free_tokens = free_groups * group_size
    
#     # First free_tokens can attend to all tokens in the first free_tokens positions
#     mask[:free_tokens, :free_tokens] = True
    
#     # For remaining tokens, create block-diagonal pattern
#     for i in range(free_tokens, seq_len, group_size):
#         end = min(i + group_size, seq_len)
#         mask[i:end, i:end] = True
    
#     return mask


# def attention_with_grouped_mask(q: Tensor, k: Tensor, v: Tensor, pe: Tensor,
#                                  group_size: int = 64, free_groups: int = 2) -> Tensor:
#     """
#     Attention with grouped masking pattern using Flash Attention.
    
#     Args:
#         q: Query tensor of shape (B, H, L, D)
#         k: Key tensor of shape (B, H, L, D)
#         v: Value tensor of shape (B, H, L, D)
#         pe: Positional encoding tensor
#         group_size: Size of attention groups (default 64)
#         free_groups: Number of initial groups without masking (default 2)
    
#     Returns:
#         Attention output of shape (B, L, H*D)
#     """
#     # Apply RoPE
#     q, k = apply_rope(q, k, pe)
    
#     B, H, L, D = q.shape
    
#     # Create the mask once and reuse for all batches/heads
#     # Using bool mask for memory efficiency
#     mask = create_grouped_attention_mask(L, group_size, free_groups, 
#                                           device=q.device, dtype=torch.bool)
    
#     # Flash Attention is automatically used when available with scaled_dot_product_attention
#     # The mask is broadcast across batch and heads dimensions
#     x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    
#     # Reshape output
#     x = rearrange(x, "B H L D -> B L (H D)")
    
#     return x


def attention_with_grouped_mask_optimized(q: Tensor, k: Tensor, v: Tensor, pe: Tensor,
                                           group_size: int = 64, free_groups: int = 2) -> Tensor:
    """
    Most optimized version: Splits computation to avoid creating masks where possible.
    
    This version is most efficient when you have many groups, as it:
    1. Processes the unrestricted attention part without any mask
    2. Processes grouped blocks separately to minimize memory usage
    """
    # Apply RoPE
    q, k = apply_rope(q, k, pe)
    
    B, H, L, D = q.shape
    free_tokens = free_groups * group_size
    
    # Initialize output tensor
    output = torch.zeros(B, H, L, D, dtype=q.dtype, device=q.device)
    
    # Part 1: Process first free_groups without any mask (most efficient)
    if free_tokens > 0:
        # No mask needed for the first part - just compute attention normally
        output[:, :, :free_tokens] = F.scaled_dot_product_attention(
            q[:, :, :free_tokens], 
            k[:, :, :free_tokens], 
            v[:, :, :free_tokens]
        )
    
    # Part 2: Process remaining groups with block-diagonal pattern
    # Process each group independently to avoid creating large masks
    for start in range(free_tokens, L, group_size):
        end = min(start + group_size, L)
        output[:, :, start:end] = F.scaled_dot_product_attention(
            q[:, :, start:end],
            k[:, :, start:end],
            v[:, :, start:end]
        )
    
    # Reshape output
    output = rearrange(output, "B H L D -> B L (H D)")
    
    return output


# Keep your original rope functions
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


# Example usage and testing
if __name__ == "__main__":
    # Setup for testing
    B, H, L, D = 4, 8, 1024, 64  # 16 groups of 64 tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test gradient flow
    print("Testing gradient flow...")
    q = torch.randn(B, H, L, D, device=device, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, requires_grad=True)
    pe = rope(torch.arange(L, device=device), D, theta=10000)
    
    # Test optimized version gradient flow
    out_opt = attention_with_grouped_mask_optimized(q, k, v, pe)
    loss_opt = out_opt.sum()
    loss_opt.backward()
    
    print(f"Gradients exist for optimized version:")
    print(f"  q.grad: {q.grad is not None}, norm: {q.grad.norm().item():.4f}")
    print(f"  k.grad: {k.grad is not None}, norm: {k.grad.norm().item():.4f}")
    print(f"  v.grad: {v.grad is not None}, norm: {v.grad.norm().item():.4f}")
    
    # Clear gradients
    q.grad = None
    k.grad = None
    v.grad = None
    
    # Test training version gradient flow with dropout
    out_train = attention_with_grouped_mask_optimized(q, k, v, pe)
    loss_train = out_train.sum()
    loss_train.backward()
    
    print(f"\nGradients exist for training version:")
    print(f"  q.grad: {q.grad is not None}, norm: {q.grad.norm().item():.4f}")
    print(f"  k.grad: {k.grad is not None}, norm: {k.grad.norm().item():.4f}")
    print(f"  v.grad: {v.grad is not None}, norm: {v.grad.norm().item():.4f}")
    

    

