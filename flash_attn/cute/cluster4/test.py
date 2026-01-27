# test_cluster4.py
"""
Test script for cluster-4 Flash Attention kernel.
"""

import torch
import time
from typing import Tuple


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation."""
    batch, seqlen_q, num_q_heads, head_dim = q.shape
    _, seqlen_k, num_kv_heads, _ = k.shape
    head_dim_v = v.shape[-1]
    
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    
    # Expand KV for GQA
    qhead_per_kvhead = num_q_heads // num_kv_heads
    if qhead_per_kvhead > 1:
        k = k.unsqueeze(3).expand(-1, -1, -1, qhead_per_kvhead, -1)
        k = k.reshape(batch, seqlen_k, num_q_heads, head_dim)
        v = v.unsqueeze(3).expand(-1, -1, -1, qhead_per_kvhead, -1)
        v = v.reshape(batch, seqlen_k, num_q_heads, head_dim_v)
    
    q = q.transpose(1, 2).float()
    k = k.transpose(1, 2).float()
    v = v.transpose(1, 2).float()
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    
    if is_causal:
        q_positions = torch.arange(seqlen_k - seqlen_q, seqlen_k, device=q.device)
        k_positions = torch.arange(seqlen_k, device=q.device)
        causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, v)
    
    o = o.transpose(1, 2).to(q.dtype)
    lse = lse.transpose(1, 2)
    
    return o, lse


def test_numerics():
    """Test numerical correctness."""
    print("=" * 60)
    print("NUMERICAL CORRECTNESS TESTS")
    print("=" * 60)
    
    try:
        from fwd_kernel_cluster4 import flash_attention_forward_cluster4
        kernel_available = True
        print("✓ Cluster-4 kernel loaded\n")
    except ImportError as e:
        print(f"✗ Kernel not available: {e}")
        print("  Using reference implementation for structure test\n")
        kernel_available = False
        flash_attention_forward_cluster4 = None
    
    test_configs = [
        # (batch, seqlen_q, seqlen_k, num_q_heads, num_kv_heads, head_dim, is_causal)
        (1, 1, 256, 8, 2, 64, False),
        (1, 1, 512, 8, 2, 64, False),
        (1, 1, 1024, 32, 8, 128, False),
        (1, 2, 1024, 32, 8, 128, False),
        (1, 1, 2048, 32, 8, 128, False),
        (1, 1, 2048, 32, 8, 128, True),  # Causal
        (8, 1, 1024, 32, 8, 128, False),  # Larger batch
    ]
    
    passed = 0
    failed = 0
    
    for batch, seqlen_q, seqlen_k, num_q_heads, num_kv_heads, head_dim, is_causal in test_configs:
        config_str = f"b={batch}, sq={seqlen_q}, sk={seqlen_k}, hq={num_q_heads}, hkv={num_kv_heads}, d={head_dim}, causal={is_causal}"
        print(f"Testing: {config_str}")
        
        # Generate inputs
        torch.manual_seed(42)
        q = torch.randn(batch, seqlen_q, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
        k = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
        v = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
        
        # Reference
        o_ref, lse_ref = attention_reference(q, k, v, is_causal=is_causal)
        
        if kernel_available:
            # Kernel under test
            try:
                o, lse = flash_attention_forward_cluster4(
                    q, k, v,
                    is_causal=is_causal,
                    cluster_shape=(4, 1),
                    return_lse=True,
                )
                
                # Compare
                o_diff = (o.float() - o_ref.float()).abs().max().item()
                lse_diff = (lse.float() - lse_ref.float()).abs().max().item()
                
                atol = 1e-2  # BF16 tolerance
                if o_diff < atol and lse_diff < atol:
                    print(f"  ✓ PASS (o_diff={o_diff:.2e}, lse_diff={lse_diff:.2e})")
                    passed += 1
                else:
                    print(f"  ✗ FAIL (o_diff={o_diff:.2e}, lse_diff={lse_diff:.2e})")
                    failed += 1
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                failed += 1
        else:
            print(f"  - SKIP (kernel not available)")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_benchmark():
    """Benchmark cluster-4 vs cluster-1 (if available)."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    try:
        from fwd_kernel_cluster4 import flash_attention_forward_cluster4
    except ImportError:
        print("Kernel not available, skipping benchmark")
        return
    
    configs = [
        # (batch, seqlen_q, seqlen_k, num_q_heads, num_kv_heads, head_dim)
        (1, 1, 1024, 32, 8, 128),
        (1, 1, 2048, 32, 8, 128),
        (1, 1, 4096, 32, 8, 128),
        (1, 2, 2048, 32, 8, 128),
        (8, 1, 2048, 32, 8, 128),
    ]
    
    warmup_iters = 5
    bench_iters = 20
    
    print(f"\n{'Config':<50} {'Cluster-1':>12} {'Cluster-4':>12} {'Speedup':>10}")
    print("-" * 90)
    
    for batch, seqlen_q, seqlen_k, num_q_heads, num_kv_heads, head_dim in configs:
        config_str = f"b={batch}, sq={seqlen_q}, sk={seqlen_k}, h={num_q_heads}/{num_kv_heads}"
        
        q = torch.randn(batch, seqlen_q, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        
        # Warmup and benchmark cluster-1
        for _ in range(warmup_iters):
            _ = flash_attention_forward_cluster4(q, k, v, cluster_shape=(1, 1))
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = flash_attention_forward_cluster4(q, k, v, cluster_shape=(1, 1))
        torch.cuda.synchronize()
        time_c1 = (time.perf_counter() - start) / bench_iters * 1000
        
        # Warmup and benchmark cluster-4
        for _ in range(warmup_iters):
            _ = flash_attention_forward_cluster4(q, k, v, cluster_shape=(4, 1))
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = flash_attention_forward_cluster4(q, k, v, cluster_shape=(4, 1))
        torch.cuda.synchronize()
        time_c4 = (time.perf_counter() - start) / bench_iters * 1000
        
        speedup = time_c1 / time_c4
        
        print(f"{config_str:<50} {time_c1:>10.3f}ms {time_c4:>10.3f}ms {speedup:>9.2f}x")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run numerical tests")
    parser.add_argument("--bench", action="store_true", help="Run benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    if args.all or (not args.test and not args.bench):
        args.test = True
        args.bench = True
    
    if args.test:
        test_numerics()
    
    if args.bench:
        test_benchmark()
```

---

## Summary: Complete File Structure

Here's what you now have:
```
flash_attn_cluster4/
├── fwd_kernel_cluster4.py      # Complete kernel (Chunks 1-11)
│   ├── class FlashAttentionForwardSm100Cluster4
│   │   ├── __init__()           # Configuration and validation
│   │   ├── _setup_*()           # Setup helpers
│   │   ├── __call__()           # Entry point (tensor setup, TMA, launch)
│   │   ├── kernel()             # Device kernel (warp dispatch)
│   │   ├── load_with_cluster()  # Load warp with multicast
│   │   ├── mma_with_cluster()   # MMA warp
│   │   ├── softmax_loop()       # Softmax warps
│   │   ├── correction_loop()    # Correction warps
│   │   └── epilogue_s2g()       # Epilogue warp
│   └── flash_attention_forward_cluster4()  # Convenience wrapper
│
└── test_cluster4.py            # Test file (Chunk 12)
    ├── attention_reference()   # PyTorch reference
    ├── test_numerics()         # Correctness tests
    └── test_benchmark()        # Performance tests
