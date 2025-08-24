import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation from scratch.
    
    Key insight: Instead of adding positional information, we rotate 
    query and key vectors by position-dependent angles.
    """
    
    def __init__(self, dim, max_seq_len=2048, base=10000):
        """
        Args:
            dim: Embedding dimension (should be even for perfect pairing)
            max_seq_len: Maximum sequence length to precompute
            base: Base for frequency computation (typically 10000)
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute rotation matrices for efficiency
        self.register_buffer('cos_cached', None, persistent=False)
        self.register_buffer('sin_cached', None, persistent=False)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        """
        Precompute cos and sin values for all positions and dimension pairs.
        
        This is the heart of RoPE - we create rotation angles that:
        1. Vary by position (m)
        2. Have different frequencies for different dimension pairs (i)
        """
        # Step 1: Create dimension pair indices
        # For dim=8: pairs are (0,1), (2,3), (4,5), (6,7)
        # We need dim//2 different frequencies
        pair_indices = torch.arange(0, self.dim, 2, dtype=torch.float32)
        
        # Step 2: Compute frequency for each dimension pair
        # Formula: 1 / (base^(2i/dim)) where i is the pair index
        # Lower indices = higher frequencies (faster rotation)
        # Higher indices = lower frequencies (slower rotation)
        freqs = 1.0 / (self.base ** (pair_indices / self.dim))
        
        # Step 3: Create position indices
        positions = torch.arange(seq_len, dtype=torch.float32)
        
        # Step 4: Compute angles for all position-frequency combinations
        # Shape: (seq_len, dim//2)
        # angles[m, i] = position_m * frequency_i
        angles = torch.outer(positions, freqs)
        print(angles)
        
        # Step 5: Duplicate angles to match original dimension
        # We need to repeat each angle twice because each pair (i, i+1) 
        # uses the same rotation angle
        # Shape changes from (seq_len, dim//2) to (seq_len, dim)
        angles = torch.repeat_interleave(angles, 2, dim=1)
        
        # Step 6: Precompute cos and sin for efficiency
        self.cos_cached = angles.cos()
        self.sin_cached = angles.sin()
    
    def _rotate_half(self, x):
        """
        Rotate the second half of the last dimension with respect to the first half.
        
        This implements the rotation operation:
        [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        
        For each pair (x_i, x_{i+1}), we get (-x_{i+1}, x_i)
        This is equivalent to multiplying by [[0, -1], [1, 0]]
        """
        # Split into two halves along the last dimension
        x1, x2 = x.chunk(2, dim=-1)
        
        # Rotate: first half becomes -second half, second half becomes first half
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_embedding(self, x, position_ids=None):
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            position_ids: Optional position indices. If None, assumes sequential positions.
            
        Returns:
            Tensor with rotary position embedding applied
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        # Handle position indices
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)
        
        # Ensure we have cached values for this sequence length
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        # Get cos and sin values for the required positions
        # Shape: (seq_len, head_dim)
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        
        # Expand dimensions to match input tensor
        # Shape: (1, seq_len, 1, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # Apply rotation using the rotation formula:
        # R(θ) * [x_even; x_odd] = [cos(θ)*x_even - sin(θ)*x_odd; sin(θ)*x_even + cos(θ)*x_odd]
        # 
        # We can rewrite this as:
        # cos(θ) * x + sin(θ) * rotate_half(x)
        x_rotated = x * cos + self._rotate_half(x) * sin
        
        return x_rotated
    
    def forward(self, query, key, position_ids=None):
        """
        Apply RoPE to query and key tensors.
        
        Args:
            query: Query tensor (batch_size, seq_len, num_heads, head_dim)
            key: Key tensor (batch_size, seq_len, num_heads, head_dim)
            position_ids: Position indices
            
        Returns:
            Rotated query and key tensors
        """
        rotated_query = self.apply_rotary_embedding(query, position_ids)
        rotated_key = self.apply_rotary_embedding(key, position_ids)
        
        return rotated_query, rotated_key


def demonstrate_rope_pairing():
    """
    Demonstrate how RoPE pairs dimensions and applies rotations.
    """
    print("=== RoPE Dimension Pairing Demonstration ===\n")
    
    # Example with small dimensions for clarity
    dim = 8
    seq_len = 20
    
    rope = RotaryPositionalEmbedding(dim=dim, max_seq_len=seq_len)
    
    print(f"Embedding dimension: {dim}")
    print(f"Number of dimension pairs: {dim // 2}")
    print(f"Pairs: {[(i, i+1) for i in range(0, dim, 2)]}\n")
    
    # Show the frequency computation
    pair_indices = torch.arange(0, dim, 2, dtype=torch.float32)
    freqs = 1.0 / (10000 ** (pair_indices / dim))
    
    print("Frequency computation:")
    for i, (pair_idx, freq) in enumerate(zip(pair_indices, freqs)):
        print(f"  Pair {i}: dimensions ({int(pair_idx)}, {int(pair_idx)+1}) -> frequency = {freq:.6f}")
    
    print(f"\nCached cos/sin shapes:")
    print(f"  cos_cached: {rope.cos_cached.shape}")
    print(f"  sin_cached: {rope.sin_cached.shape}")
    
    # Show how angles are duplicated
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    angles_duplicated = torch.repeat_interleave(angles, 2, dim=1)
    
    print(f"\nAngle tensor evolution:")
    print(f"  Original angles shape (pos × pairs): {angles.shape}")
    print(f"  After duplication (pos × dim): {angles_duplicated.shape}")
    
    print(f"\nFirst position angles (before duplication):")
    print(f"  {angles[0].tolist()}")
    print(f"First position angles (after duplication):")
    print(f"  {angles_duplicated[0].tolist()}")
    
    # Demonstrate rotation on sample data
    print(f"\n=== Sample Rotation ===")
    
    # Create sample query tensor
    batch_size, num_heads, head_dim = 1, 2, dim
    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    print(f"Input shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    
    # Apply RoPE
    rotated_query, rotated_key = rope(query, key)
    
    print(f"\nAfter RoPE:")
    print(f"  Rotated Query: {rotated_query.shape}")
    print(f"  Rotated Key: {rotated_key.shape}")
    
    # Show the effect on first token, first head
    original = query[0, 0, 0, :4]  # First 4 dimensions
    rotated = rotated_query[0, 0, 0, :4]
    
    print(f"\nExample transformation (first token, first head, first 4 dims):")
    print(f"  Original: {original.tolist()}")
    print(f"  Rotated:  {rotated.tolist()}")


def test_rope_properties():
    """
    Test key properties of RoPE implementation.
    """
    print("\n=== Testing RoPE Properties ===\n")
    
    dim = 64
    seq_len = 10
    rope = RotaryPositionalEmbedding(dim=dim, max_seq_len=seq_len)
    
    # Test 1: Rotation preserves vector norm
    x = torch.randn(1, seq_len, 1, dim)
    x_rotated = rope.apply_rotary_embedding(x)
    
    original_norms = torch.norm(x, dim=-1)
    rotated_norms = torch.norm(x_rotated, dim=-1)
    
    print("Test 1 - Norm preservation:")
    print(f"  Max norm difference: {torch.max(torch.abs(original_norms - rotated_norms)).item():.10f}")
    print(f"  Norms preserved: {torch.allclose(original_norms, rotated_norms, atol=1e-6)}")
    
    # Test 2: Same position gives same rotation
    pos_0_a = rope.apply_rotary_embedding(x[:, :1])  # Position 0
    pos_0_b = rope.apply_rotary_embedding(x[:, :1])  # Position 0 again
    
    print(f"\nTest 2 - Consistent rotation:")
    print(f"  Same position rotations match: {torch.allclose(pos_0_a, pos_0_b)}")
    
    # Test 3: Different positions give different rotations
    pos_0 = rope.apply_rotary_embedding(x[:, :1])
    pos_5 = rope.apply_rotary_embedding(x[:, 5:6])
    
    print(f"\nTest 3 - Different positions:")
    print(f"  Different positions give different results: {not torch.allclose(pos_0, pos_5)}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_rope_pairing()
    test_rope_properties()
    
    print("\n=== Usage Example ===")
    
    # Typical usage in a transformer
    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
    
    # Initialize RoPE
    rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=2048)
    
    # Sample query and key tensors (as they would come from linear projections)
    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # Apply rotary embedding
    rotated_query, rotated_key = rope(query, key)
    
    # Now use rotated_query and rotated_key in attention computation
    print(f"Ready for attention computation with shapes:")
    print(f"  Query: {rotated_query.shape}")
    print(f"  Key: {rotated_key.shape}")