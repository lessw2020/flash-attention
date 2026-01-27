# ========== Continuing FlashAttentionForwardSm100Cluster4 class ==========
    
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        cta_id_in_cluster: Int32,  # NEW: cluster position
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        blocksparse_tensors=None,
    ):
        """
        Softmax loop with cluster awareness.
        
        Each CTA computes softmax on its own S matrix (different Q blocks).
        The cluster info is used to compute the correct M-block index.
        """
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        
        # ========== Setup TMEM access patterns ==========
        tStScale = cute.composition(tStSi, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        
        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width
        tStP_layout = cute.composition(tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32)))
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)
        
        # TMEM copy atoms
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)
        
        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32,
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(tidx)
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)
        
        # ========== Phase tracking ==========
        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)
        
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset + warp_idx_in_wg
        
        # ========== Main loop ==========
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        
        while work_tile.is_valid_tile:
            m_block_base, head_idx, batch_idx, split_idx = work_tile.tile_idx
            
            # ========== Cluster M-block adjustment ==========
            m_block = m_block_base * self.cluster_m + cta_id_in_cluster
            
            seqlen = SeqlenInfoCls(batch_idx)
            
            # Check if this CTA has valid work
            seqlen_q_tiles = cute.ceil_div(seqlen.seqlen_q, self.m_block_size * self.q_stage)
            has_valid_work = (m_block < seqlen_q_tiles)
            
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            
            # ========== Create mask function ==========
            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                m_block=self.q_stage * m_block + stage,  # Uses cluster-adjusted m_block
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_tensors=aux_tensors,
            )
            
            # Recompute fastdiv_mods if needed
            if cutlass.const_expr(aux_tensors is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                if seqlen.has_cu_seqlens_q or seqlen.has_seqused_q:
                    seqlen_q_divmod = FastDivmodDivisor(seqlen.seqlen_q)
                if seqlen.has_cu_seqlens_k or seqlen.has_seqused_k:
                    seqlen_k_divmod = FastDivmodDivisor(seqlen.seqlen_k)
                fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)
            
            mask_mod = self.mask_mod if const_expr(self.mask_mod is not None) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=mask_mod,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                **shared_mask_kwargs,
            )
            
            # ========== Initialize softmax state ==========
            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()
            
            tile_block_count = n_block_max - n_block_min
            has_work = (const_expr(not self.is_split_kv) or tile_block_count > Int32(0)) and has_valid_work
            
            softmax_step = partial(
                self._softmax_step,
                softmax=softmax,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=self.q_stage * m_block + stage,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
            )
            
            if has_work:
                # Wait for correction to signal empty
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, 
                    si_corr_producer_phase
                )
                si_corr_producer_phase ^= 1
                
                # Process all N blocks
                if const_expr(not self.is_split_kv) or tile_block_count > Int32(0):
                    # First block with seqlen masking
                    mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                    )
                    n_block_max -= 1
                    
                    # Blocks with causal/local masking
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                        for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                    
                    # Remaining blocks (no causal mask needed)
                    n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                        n_block = n_block_max - n_tile - 1
                        if const_expr(self.mask_mod is not None):
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                                mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        else:
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                                mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block,
                            )
                    
                    # Local masking on the left
                    if const_expr(self.is_local and block_info.window_size_left is not None):
                        n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                        for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                    
                    # Write final scale to SMEM
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[tidx + stage * self.m_block_size + self.m_block_size * 2] = softmax.row_max[0]
                    
                    # Signal correction warp
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)
            
            # Advance to next cluster work unit
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
    
    @cute.jit
    def _softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors=None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Single softmax step - same as base kernel."""
        
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tScP = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32)))
        
        # Wait for S from MMA
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        
        # Load S from TMEM
        tSrS_t2r = cute.make_fragment(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        
        # Apply score mod if present
        if cutlass.const_expr(self.score_mod is not None):
            self._apply_score_mod(
                tSrS_t2r, thr_tmem_load, thr_mma_qk,
                batch_idx, head_idx, m_block, n_block, softmax,
                seqlen, aux_tensors, fastdiv_mods, head_divmod,
            )
        
        # Apply mask
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        
        # Update row max
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
        
        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        
        # Signal correction that row_max is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)
        
        # Subtract max and apply exp2
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        
        # Sequence barrier
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_wait(
                mbar_ptr + mbar_s0_s1_sequence_offset + stage * 4, 
                s0_s1_sequence_phase
            )
        
        # Convert to P
        tSrP_r2t_f32 = cute.make_fragment(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),
            tSrS_t2r.layout,
        )
        softmax.apply_exp2_convert(
            tSrS_t2r, tSrP_r2t,
            e2e=mask_fn is None and self.head_dim_padded <= 128,
            e2e_freq=self.e2e_freq,
        )
        
        # Sequence barrier
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_s0_s1_sequence_offset + (1 - stage) * 4)
        
        # Store P to TMEM
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2]) // 4 * 3):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        
        # Signal MMA that P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        
        # Store remaining P
        for i in cutlass.range_constexpr(
            cute.size(tStP_r2t.shape[2]) // 4 * 3, cute.size(tStP_r2t.shape[2])
        ):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        
        # Signal second half ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        
        # Wait for correction to read previous scale
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, 
            si_corr_producer_phase
        )
        
        # Update row sum
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1
