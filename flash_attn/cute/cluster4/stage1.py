# fwd_kernel_cluster4.py
"""
Flash Attention Forward Kernel with Cluster-4 Support for SM100 (Blackwell).

This kernel extends the base SM100 Flash Attention implementation to support
cluster_shape=(4,1) for TMA multicast optimization, reducing KV memory traffic
by 4x through hardware multicast.

Key modifications from base kernel:
1. TMA multicast for K, V loads (only leader CTA issues loads)
2. Cluster-level synchronization barriers
3. Modified tile scheduler for cluster-coherent work assignment
4. Adjusted pipeline for cluster sync overhead

Optimized for decode workloads (seqlen_q=1,2) with GQA.

Author: [Your name]
Date: [Date]
"""

import enum
import math
from typing import Type, Tuple, Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic

# Import from flash_attn package
from flash_attn.cute.paged_kv import PagedKVManager
import flash_attn.cute.utils as utils
from flash_attn.cute import copy_utils
import flash_attn.cute.pipeline as pipeline
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import SoftmaxSm100, apply_score_mod_inner
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.pack_gqa import PackGQA
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from cutlass.cute import FastDivmodDivisor
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)


class NamedBarrierFwdCluster(enum.IntEnum):
    """Named barriers for cluster-4 kernel."""
    Epilogue = enum.auto()
    ClusterKVSync = enum.auto()  # New: cluster-level KV synchronization


class FlashAttentionForwardSm100Cluster4:
    """
    Flash Attention Forward Kernel with Cluster-4 Multicast Support.
    
    This kernel processes multiple M-blocks (Q tiles) in a cluster, where all
    CTAs in the cluster share the same K,V data via TMA multicast. This reduces
    global memory bandwidth for KV by a factor equal to cluster_m.
    
    Cluster layout:
        cluster_shape_mn = (4, 1) means 4 CTAs along M dimension
        - CTA 0: processes M-block 0, 4, 8, ...
        - CTA 1: processes M-block 1, 5, 9, ...
        - CTA 2: processes M-block 2, 6, 10, ...
        - CTA 3: processes M-block 3, 7, 11, ...
        All CTAs process the SAME sequence of N-blocks (KV).
    
    Memory traffic reduction:
        Without cluster: 4 CTAs each load K,V = 4x traffic
        With cluster: 1 CTA loads K,V, multicast to 4 = 1x traffic
    """
    
    arch = 100
    
    # Cluster configuration
    MAX_CLUSTER_SIZE = 16  # Hardware limit
    SUPPORTED_CLUSTER_M = (1, 2, 4)  # Tested configurations
    
    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int | None = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        cluster_shape_mn: Tuple[int, int] = (4, 1),  # Default to cluster-4
        q_stage: cutlass.Constexpr[int] = 2,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
    ):
        """
        Initialize the Cluster-4 Flash Attention kernel.
        
        Args:
            head_dim: Dimension of K (and Q) heads
            head_dim_v: Dimension of V heads (defaults to head_dim)
            qhead_per_kvhead: Number of Q heads per KV head (for GQA)
            is_causal: Whether to apply causal masking
            is_local: Whether to use sliding window attention
            is_split_kv: Whether to split KV across multiple passes
            pack_gqa: Whether to pack multiple Q heads into one tile
            m_block_size: Tile size for M dimension (Q sequence)
            n_block_size: Tile size for N dimension (KV sequence)
            cluster_shape_mn: Cluster dimensions (cluster_m, cluster_n)
            q_stage: Pipeline depth for Q (1 or 2)
            is_persistent: Whether to use persistent kernel scheduling
            score_mod: Optional score modification function
            mask_mod: Optional mask modification function
        """
        
        # ========== Cluster Configuration ==========
        self.cluster_m, self.cluster_n = cluster_shape_mn
        self._validate_cluster_config()
        
        # Multicast is beneficial when cluster_m > 1
        self.use_multicast_kv = self.cluster_m > 1
        
        # For paged KV, we currently don't support multicast (would need gather)
        self.use_tma_KV = not paged_kv_non_tma
        if paged_kv_non_tma and self.use_multicast_kv:
            raise NotImplementedError(
                "Cluster multicast not yet supported with paged KV cache. "
                "Use cluster_shape_mn=(1,1) for paged KV."
            )
        
        # ========== Head Dimension Setup ==========
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        
        # ========== Tiling Configuration ==========
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = q_stage
        assert self.q_stage in [1, 2], f"q_stage must be 1 or 2, got {q_stage}"
        
        # CTA tiler: each CTA processes q_stage M-blocks
        self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (m_block_size, self.head_dim_v_padded, n_block_size)
        
        # ========== Accumulator Configuration ==========
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        
        # ========== Feature Flags ==========
        self.cluster_shape_mn = cluster_shape_mn
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.use_correction_warps_for_epi = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        self.q_subtile_factor = q_subtile_factor
        
        # Validate PackGQA config
        if pack_gqa:
            assert m_block_size % self.qhead_per_kvhead == 0, (
                f"For PackGQA, m_block_size ({m_block_size}) must be divisible "
                f"by qhead_per_kvhead ({self.qhead_per_kvhead})"
            )
        
        # Split KV not supported for large hdim
        assert not (self.is_split_kv and self.head_dim_v_padded >= 192), (
            "SplitKV is not supported for hdim >= 192"
        )
        
        # ========== Score/Mask Modifications ==========
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 2
        
        # ========== Pipeline Configuration ==========
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = (
            (self.head_dim_padded == 192 and self.head_dim_v_padded >= 64) or
            (self.head_dim_v_padded >= 128 and self.is_split_kv)
        )
        if self.overlap_sO_sQ:
            self.is_persistent = False
        
        # ========== Warp Assignment ==========
        self._setup_warp_assignment(paged_kv_non_tma)
        
        # ========== TMEM Configuration ==========
        self._setup_tmem_layout()
        
        # ========== Register Allocation ==========
        self._setup_register_allocation(paged_kv_non_tma)
        
        # ========== Misc ==========
        self.buffer_align_bytes = 1024
    
    def _validate_cluster_config(self):
        """Validate cluster configuration."""
        cluster_size = self.cluster_m * self.cluster_n
        
        if cluster_size > self.MAX_CLUSTER_SIZE:
            raise ValueError(
                f"Cluster size {self.cluster_m}x{self.cluster_n}={cluster_size} "
                f"exceeds maximum {self.MAX_CLUSTER_SIZE}"
            )
        
        if self.cluster_m not in self.SUPPORTED_CLUSTER_M:
            raise ValueError(
                f"cluster_m={self.cluster_m} not in supported values {self.SUPPORTED_CLUSTER_M}. "
                f"Other values may work but are not tested."
            )
        
        # Cluster dimensions must be powers of 2
        if self.cluster_m & (self.cluster_m - 1) != 0:
            raise ValueError(f"cluster_m must be power of 2, got {self.cluster_m}")
        if self.cluster_n & (self.cluster_n - 1) != 0:
            raise ValueError(f"cluster_n must be power of 2, got {self.cluster_n}")
    
    def _setup_warp_assignment(self, paged_kv_non_tma: bool):
        """Configure warp specialization."""
        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )
        
        # Adjust warp roles based on q_stage and TMA usage
        if self.q_stage == 1:
            if not self.use_tma_KV:
                self.empty_warp_ids = self.empty_warp_ids + self.load_warp_ids
                self.load_warp_ids = self.softmax1_warp_ids
            else:
                self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        elif not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()
        
        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids
        elif self.is_varlen_q:
            self.epilogue_warp_ids = (13, 14)
    
    def _setup_tmem_layout(self):
        """Configure Tensor Memory layout."""
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        
        # S matrix offsets (two stages for double buffering)
        self.tmem_s_offset = [0, self.n_block_size]  # 0, 128
        
        # O accumulator offsets
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # 256, 384
        
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS, (
            f"TMEM usage {self.tmem_total} exceeds capacity {SM100_TMEM_CAPACITY_COLUMNS}"
        )
        
        # P matrix overlays part of S (after softmax converts S to P)
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 64, 192
        
        # Vector buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset
    
    def _setup_register_allocation(self, paged_kv_non_tma: bool):
        """Configure register allocation per warp type."""
        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200 if not paged_kv_non_tma else 184
            self.num_regs_correction = 64
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
        else:
            self.num_regs_softmax = 200 if not paged_kv_non_tma else 184
            self.num_regs_correction = 64
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
        
        self.num_regs_empty = 24
