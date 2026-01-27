# ========== Continuing FlashAttentionForwardSm100Cluster4 class ==========
    
    @cute.jit
    def load_with_cluster(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        multicast_mask: Int32,
        cta_id_in_cluster: Int32,
        is_kv_load_leader: bool,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors,
    ):
        """
        Load warp function with cluster multicast support.
        
        Key differences from base kernel:
        1. Each CTA loads its own Q (different M-blocks)
        2. Only leader CTA loads K,V with multicast to all cluster CTAs
        3. Cluster synchronization ensures all CTAs receive KV before proceeding
        
        Multicast flow:
        - Leader CTA issues TMA load with multicast_mask
        - Hardware delivers data to all CTAs' SMEM simultaneously
        - All CTAs wait on their local barrier for data arrival
        """
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        
        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.kv_stage
        )
        
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        
        # ========== Cluster-aware work assignment ==========
        # Each CTA in cluster gets a different M-block offset
        # CTA 0: m_block 0, 4, 8, ...
        # CTA 1: m_block 1, 5, 9, ...
        # etc.
        
        while work_tile.is_valid_tile:
            m_block_base, head_idx, batch_idx, split_idx = work_tile.tile_idx
            
            # ========== Cluster M-block adjustment ==========
            # The tile scheduler gives us the "cluster work unit"
            # Each CTA processes: m_block_base * cluster_m + cta_id_in_cluster
            m_block = m_block_base * self.cluster_m + cta_id_in_cluster
            
            # Check if this CTA has valid work (may be beyond bounds)
            seqlen = SeqlenInfoCls(batch_idx)
            seqlen_q_tiles = cute.ceil_div(seqlen.seqlen_q, self.m_block_size * self.q_stage)
            has_valid_work = (m_block < seqlen_q_tiles)
            
            # ========== Setup Q tensor for this CTA's M-block ==========
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0))
            
            # ========== Setup K,V tensors (same for all CTAs in cluster) ==========
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
                else:
                    mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
                    mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            else:
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None))
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None))
            
            # ========== Create TMA partitions ==========
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )
            
            if const_expr(self.use_tma_KV):
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K, 0, cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V, 0, cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                paged_kv_manager = None
            else:
                paged_kv_manager = None  # Not supported with cluster
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None
            
            # ========== Create load functions ==========
            load_Q = partial(
                self._load_Q_single_cta,
                load_Q_fn,
                mbar_ptr + self.mbar_load_q_full_offset,
                mbar_ptr + self.mbar_load_q_empty_offset,
                phase=q_producer_phase,
            )
            
            # K,V load with multicast
            load_K = partial(
                self._load_KV_multicast,
                tma_atom_K,
                tKgK,
                tKsK,
                sK,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                multicast_mask,
                is_kv_load_leader,
                K_or_V="K",
            )
            
            load_V = partial(
                self._load_KV_multicast,
                tma_atom_V,
                tVgV,
                tVsV,
                sV,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                multicast_mask,
                is_kv_load_leader,
                K_or_V="V",
            )
            
            # ========== Get N-block range (same for all CTAs in cluster) ==========
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            
            # ========== Issue loads ==========
            if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                # Only load Q if this CTA has valid work
                if has_valid_work:
                    if const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE:
                        load_Q(block=self.q_stage * m_block + 0, stage=0)
                
                # KV loads - all CTAs participate in sync, but only leader loads
                n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                page_idx = (
                    mPageTable[batch_idx, n_block_first]
                    if const_expr(mPageTable is not None and self.use_tma_KV)
                    else None
                )
                
                # First K load
                load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)
                kv_producer_state.advance()
                
                # Second Q load
                if has_valid_work:
                    if const_expr(self.q_stage == 2):
                        if const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE:
                            load_Q(block=self.q_stage * m_block + 1, stage=1)
                
                q_producer_phase ^= 1
                
                # First V load
                load_V(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)
                kv_producer_state.advance()
                
                # Remaining KV blocks
                for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                    n_block = n_block_max - 2 - i
                    page_idx = (
                        mPageTable[batch_idx, n_block]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    
                    load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
                    kv_producer_state.advance()
                    load_V(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
                    kv_producer_state.advance()
            
            # ========== Advance to next cluster work unit ==========
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
    
    def _load_Q_single_cta(
        self,
        load_Q_fn: Callable,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        """Load Q - each CTA loads its own Q tile (no multicast)."""
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, self.tma_copy_bytes["Q"])
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)
    
    @cute.jit
    def _load_KV_multicast(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        sX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        multicast_mask: Int32,
        is_kv_load_leader: bool,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
    ):
        """
        Load K or V with TMA multicast to all CTAs in cluster.
        
        Only the leader CTA issues the TMA load. The hardware multicast
        delivers data to all CTAs' shared memory simultaneously.
        
        All CTAs (including non-leaders) must:
        1. Wait for their local buffer to be empty
        2. Signal expected bytes on their local barrier
        3. Wait for data to arrive
        
        This ensures proper synchronization across the cluster.
        """
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        
        # ========== All CTAs wait for buffer empty ==========
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        
        # Handle uneven KV SMEM layout
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            if stage == 0:
                cute.arch.mbarrier_wait(mbar_empty_ptr + 1, phase)
        
        # ========== Leader CTA: issue multicast TMA ==========
        if is_kv_load_leader:
            with cute.arch.elect_one():
                # Signal expected bytes to local barrier
                # With multicast, hardware will signal all CTAs' barriers
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_full_ptr + stage, 
                    self.tma_copy_bytes[K_or_V],
                )
            
            # Issue TMA with multicast
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                tXsX_cur = self._offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            
            tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
            
            # The key: TMA copy with multicast_mask
            # This delivers data to all CTAs specified in the mask
            cute.copy(
                tma_atom, 
                tXgX_cur, 
                tXsX_cur, 
                tma_bar_ptr=mbar_full_ptr + stage,
                multicast_mask=multicast_mask,  # Multicast to all cluster CTAs
            )
        else:
            # ========== Non-leader CTAs: just signal expected bytes ==========
            # They'll receive data via multicast
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_full_ptr + stage,
                    self.tma_copy_bytes[K_or_V],
                )
    
    @cute.jit
    def _offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        """Adjust SMEM pointer for uneven K/V head dimensions."""
        if const_expr(self.uneven_kv_smem):
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX
