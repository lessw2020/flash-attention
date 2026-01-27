# ========== Continuing FlashAttentionForwardSm100Cluster4 class ==========
    
    def _setup_attributes(self):
        """Set up configurations after dtype is known."""
        self.kv_stage = 4 if self.q_dtype.width == 8 or self.q_stage == 1 else 3
        self.acc_stage = 1
        
        # Handle uneven K/V head dimensions for SMEM layout
        self.uneven_kv_smem = (
            self.head_dim_padded == 192 and 
            self.head_dim_v_padded == 128 and 
            self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.m_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0
    
    def _setup_barrier_offsets(self):
        """Configure barrier offsets for pipeline synchronization."""
        # Standard barriers (same as base kernel)
        self.mbar_load_q_full_offset = 0
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + self.q_stage
        self.mbar_load_kv_full_offset = self.mbar_load_q_empty_offset + self.q_stage
        self.mbar_load_kv_empty_offset = self.mbar_load_kv_full_offset + self.kv_stage
        self.mbar_P_full_O_rescaled_offset = self.mbar_load_kv_empty_offset + self.kv_stage
        self.mbar_S_full_offset = self.mbar_P_full_O_rescaled_offset + self.q_stage
        self.mbar_O_full_offset = self.mbar_S_full_offset + self.q_stage
        self.mbar_softmax_corr_full_offset = self.mbar_O_full_offset + self.q_stage
        self.mbar_softmax_corr_empty_offset = self.mbar_softmax_corr_full_offset + self.q_stage
        self.mbar_corr_epi_full_offset = self.mbar_softmax_corr_empty_offset + self.q_stage
        self.mbar_corr_epi_empty_offset = self.mbar_corr_epi_full_offset + self.q_stage
        self.mbar_s0_s1_sequence_offset = self.mbar_corr_epi_empty_offset + self.q_stage
        self.mbar_tmem_dealloc_offset = self.mbar_s0_s1_sequence_offset + 8
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1
        
        # ===== NEW: Cluster-specific barriers =====
        # These are used for cluster-level synchronization of KV loads
        self.mbar_cluster_kv_ready_offset = self.mbar_P_full_2_offset + self.q_stage
        self.mbar_cluster_kv_consumed_offset = self.mbar_cluster_kv_ready_offset + self.kv_stage
        
        # Total barrier count
        self.mbar_total = self.mbar_cluster_kv_consumed_offset + self.kv_stage
    
    def _compute_multicast_mask(self) -> int:
        """
        Compute the multicast mask for TMA operations.
        
        The multicast mask indicates which CTAs in the cluster should receive
        the multicast data. For cluster_m=4, all 4 CTAs along M receive the data.
        
        Returns:
            Bitmask where bit i is set if CTA i should receive the data.
        """
        # For cluster_shape_mn = (4, 1):
        # CTAs are numbered 0, 1, 2, 3 along M
        # All should receive KV data: mask = 0b1111 = 15
        return (1 << self.cluster_m) - 1
    
    def _compute_smem_sizes(self, sQ_layout, sK_layout, sO_layout):
        """Compute shared memory sizes for allocation."""
        sO_size = cute.cosize(sO_layout) if not self.overlap_sO_sQ else 0
        sQ_size = (
            cute.cosize(sQ_layout) if not self.overlap_sO_sQ else
            cutlass.max(
                cute.cosize(sQ_layout), 
                cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width
            )
        )
        sK_size = cute.cosize(sK_layout)
        
        return sO_size, sQ_size, sK_size
    
    def _get_cluster_cta_rank(self) -> Tuple[int, int]:
        """
        Get this CTA's position within the cluster.
        
        Returns:
            (cta_m, cta_n): Position in M and N dimensions of cluster
        """
        # On SM100, cluster CTA ID encodes position
        # For cluster_shape=(4,1): cta_id 0-3 corresponds to cta_m 0-3
        cta_id = cute.arch.cluster_cta_id()
        cta_m = cta_id % self.cluster_m
        cta_n = cta_id // self.cluster_m
        return cta_m, cta_n
    
    def _is_kv_load_leader(self, cta_id_in_cluster: Int32) -> bool:
        """
        Determine if this CTA is responsible for KV loads.
        
        For multicast, only one CTA (the leader) issues TMA loads.
        Other CTAs receive data via hardware multicast.
        
        Args:
            cta_id_in_cluster: This CTA's ID within the cluster
            
        Returns:
            True if this CTA should issue KV loads
        """
        # CTA 0 is always the leader
        return cta_id_in_cluster == 0
