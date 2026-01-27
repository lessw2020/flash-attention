# ========== End of FlashAttentionForwardSm100Cluster4 class ==========


def flash_attention_forward_cluster4(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    softmax_scale: float = None,
    is_causal: bool = False,
    window_size_left: int = None,
    window_size_right: int = None,
    cluster_shape: Tuple[int, int] = (4, 1),
    return_lse: bool = False,
) -> "torch.Tensor":
    """
    Run Flash Attention with cluster-4 multicast optimization.
    
    This function provides a convenient interface to the cluster-4 kernel,
    optimized for decode workloads (small seqlen_q) with GQA.
    
    Args:
        q: Query tensor of shape (batch, seqlen_q, num_q_heads, head_dim)
        k: Key tensor of shape (batch, seqlen_k, num_kv_heads, head_dim)  
        v: Value tensor of shape (batch, seqlen_k, num_kv_heads, head_dim_v)
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking
        window_size_left: Left window size for local attention (None = unlimited)
        window_size_right: Right window size for local attention (None = unlimited)
        cluster_shape: Cluster dimensions (cluster_m, cluster_n), default (4, 1)
        return_lse: Whether to return log-sum-exp values
        
    Returns:
        output: Attention output of shape (batch, seqlen_q, num_q_heads, head_dim_v)
        lse: (optional) Log-sum-exp of shape (batch, seqlen_q, num_q_heads) if return_lse=True
        
    Performance notes:
        - Best for decode (seqlen_q=1 or 2) with long context (seqlen_k >= 1024)
        - cluster_shape=(4,1) reduces KV memory traffic by ~4x via TMA multicast
        - Automatically enables PackGQA when num_q_heads > num_kv_heads
    """
    import torch
    
    # Validate inputs
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA"
    assert q.dtype == k.dtype == v.dtype, "All inputs must have same dtype"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only FP16 and BF16 supported"
    
    batch, seqlen_q, num_q_heads, head_dim = q.shape
    _, seqlen_k, num_kv_heads, _ = k.shape
    head_dim_v = v.shape[-1]
    
    # Validate GQA configuration
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    qhead_per_kvhead = num_q_heads // num_kv_heads
    
    # Default softmax scale
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    
    # Determine if we should use PackGQA
    use_pack_gqa = qhead_per_kvhead > 1
    
    # Allocate output
    o = torch.empty(
        (batch, seqlen_q, num_q_heads, head_dim_v),
        dtype=q.dtype,
        device=q.device
    )
    
    lse = None
    if return_lse:
        lse = torch.empty(
            (batch, seqlen_q, num_q_heads),
            dtype=torch.float32,
            device=q.device
        )
    
    # Get CUDA stream
    stream = torch.cuda.current_stream().cuda_stream
    
    # Create kernel instance
    kernel = FlashAttentionForwardSm100Cluster4(
        head_dim=head_dim,
        head_dim_v=head_dim_v if head_dim_v != head_dim else None,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=is_causal,
        is_local=(window_size_left is not None or window_size_right is not None),
        cluster_shape_mn=cluster_shape,
        pack_gqa=use_pack_gqa,
    )
    
    # Convert to CuTe tensors
    mQ = cute.from_dlpack(q.detach())
    mK = cute.from_dlpack(k.detach())
    mV = cute.from_dlpack(v.detach())
    mO = cute.from_dlpack(o)
    mLSE = cute.from_dlpack(lse) if lse is not None else None
    
    # Run kernel
    kernel(
        mQ, mK, mV, mO, mLSE,
        softmax_scale=Float32(softmax_scale),
        stream=cuda.CUstream(stream),
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    
    if return_lse:
        return o, lse
    return o


def can_use_cluster4(
    seqlen_q: int,
    seqlen_k: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[bool, str]:
    """
    Check if cluster-4 optimization is beneficial for given configuration.
    
    Returns:
        (can_use, reason): Tuple of (bool, explanation string)
    """
    # Check basic requirements
    if num_q_heads % num_kv_heads != 0:
        return False, f"num_q_heads ({num_q_heads}) not divisible by num_kv_heads ({num_kv_heads})"
    
    if head_dim not in [64, 96, 128, 192]:
        return False, f"head_dim ({head_dim}) not in supported values [64, 96, 128, 192]"
    
    # Check if there's enough work for cluster
    total_q_tiles = (seqlen_q + 127) // 128  # Assuming m_block_size=128
    if total_q_tiles < 4:
        # With cluster_m=4, we need at least 4 Q tiles to fully utilize
        # Still works but may not be beneficial
        pass
    
    # Check if KV is long enough to benefit from multicast
    if seqlen_k < 256:
        return False, f"seqlen_k ({seqlen_k}) too short to benefit from multicast"
    
    # Estimate benefit
    # Multicast saves ~3x KV loads (4 CTAs share instead of loading separately)
    # But adds cluster sync overhead
    estimated_kv_bytes = 2 * seqlen_k * num_kv_heads * head_dim * 2  # K + V, assuming BF16
    
    if estimated_kv_bytes < 64 * 1024:  # 64KB threshold
        return True, "Marginal benefit (KV cache small)"
    
    return True, f"Good candidate: {estimated_kv_bytes / 1024 / 1024:.1f}MB KV cache"
