import torch
import torch.nn.functional as F
import math

def sliding_window_attention_scores(Q, K, window_size):
    """
    Computes attention scores for a sliding window attention mechanism efficiently.

    This function follows the logic described in the provided text:
    1. Pads the Key tensor to handle edge cases for the window.
    2. Creates a "windowed" Key tensor (Kw) using tensor unfolding.
    3. Computes the alignment scores (E) using a single, efficient tensor operation (einsum).

    Args:
        Q (torch.Tensor): The Query tensor of shape (nhead, dhead, N).
        K (torch.Tensor): The Key tensor of shape (nhead, dhead, N).
        window_size (int): The size of the sliding window (w).

    Returns:
        torch.Tensor: The alignment scores tensor E of shape (nhead, N, w).
    """
    # Get dimensions from the input tensors
    nhead, dhead, N = Q.shape
    w = window_size

    # --- Step 1: Pad the keys to account for the edge cases ---
    # As described in Fig. 17, we pad the keys on the left.
    # A window of size 'w' needs 'w-1' previous keys to be available for the first query.
    # The padding is applied only to the sequence dimension (the last dimension).
    padding = (w - 1, 0)  # (pad_left, pad_right) for the last dimension
    K_padded = F.pad(K, padding, "constant", 0)
    # Shape of K_padded is now: (nhead, dhead, N + w - 1)

    # --- Step 2: Generate the windowed keys tensor Kw ---
    # This is the crucial step to create the tensor for parallel computation.
    # We use `unfold` to create a sliding window view of the padded keys.
    # This is a highly efficient, O(N) operation that avoids Python loops.
    # .unfold(dimension, size, step)
    Kw = K_padded.unfold(2, w, 1)
    # The resulting shape is exactly as described in the text: (nhead, dhead, N, w)

    # --- Step 3: Compute the product between Kw and Q ---
    # We use `torch.einsum` as it's the most explicit and efficient way to perform
    # the specified tensor contraction: E_hnw = Σ_d (Q_hdn * Kw_hdnw)
    # 'h' = nhead, 'd' = dhead, 'n' = sequence length N, 'w' = window_size
    einsum_formula = "hdn,hdnw->hnw"
    E = torch.einsum(einsum_formula, Q, Kw)

    # --- Step 4: Scale by the square root of d_head ---
    # This is the standard scaling factor in attention mechanisms.
    E_scaled = E / math.sqrt(dhead)

    # The final shape is (nhead, N, w), as specified in the text.
    return E_scaled


if __name__=="__main__":

    # Define the dimensions of our tensors
    N = 1024      # Sequence length
    d_model = 512   # Model dimension
    nhead = 8       # Number of attention heads
    dhead = d_model // nhead # Dimension per head
    w = 4         # Window size, as in the example figure

    # Create dummy Query and Key tensors with the specified shape
    # Note: The text uses (nhead, dhead, N), which is common in some frameworks.
    # We will stick to this convention.
    Q = torch.randn(nhead, dhead, N)
    K = torch.randn(nhead, dhead, N)
    V = torch.randn(nhead, dhead, N)

    # Compute the alignment scores
    E = sliding_window_attention_scores(Q, K, window_size=w)

    # --- Verification ---
    print(f"Input Q shape: {Q.shape}")
    print(f"Input K shape: {K.shape}")
    print("-" * 30)
    print(f"Specified window size (w): {w}")
    print(f"Final alignment scores tensor E shape: {E.shape}")
    print(f"Expected final shape: ({nhead}, {N}, {w})")

    # Check if the output shape matches the expected shape
    # E has shape (nhead, N, w)
    assert E.shape == (nhead, N, w)

    # attention_weights also has shape (nhead, N, w)
    attention_weights = F.softmax(E, dim=-1)


    # Assume V has shape (nhead, dhead, N)
    # Pad V in the same way as K
    padding = (w - 1, 0) 
    V_padded = F.pad(V, padding, "constant", 0)
    # Unfold V to get the windowed version
    Vw = V_padded.unfold(2, w, 1)
    # Vw now has the shape (nhead, dhead, N, w)

    # We can again use einsum for an efficient and clear operation
    # attention_weights shape: (nhead, N, w)
    # Vw shape: (nhead, dhead, N, w)
    output = torch.einsum("hnw,hdnw->hdn", attention_weights, Vw)
    # The final output has shape (nhead, dhead, N)

    # Reshape to combine heads
    # output shape: (nhead, dhead, N) -> (N, nhead, dhead) -> (N, nhead * dhead)
    output_reshaped = output.permute(2, 0, 1).contiguous().view(N, nhead * dhead)
