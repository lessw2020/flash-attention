# ========== Continuing FlashAttentionForwardSm100Cluster4 class ==========
    
    @cute.jit
    def mma_with_cluster(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        cta_id_in_cluster: Int32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors,
    ):
        """
        MMA warp function with cluster awareness.
        
        Each CTA in the cluster:
        - Has its own Q in SMEM (different M-blocks)
        - Shares the same K,V data (via multicast)
        - Computes its own S = Q × K^T and O = P × V
        
        The MMA operations are identical to the base kernel;
        the cluster optimization is entirely in the data loading.
        """
        
        # ========== Create MMA fragments ==========
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0],)
        
        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        
        # ========== Create GEMM functions ==========
        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_partial,
                qk_mma_op,
                self.tmem_s_offset[stage],
                tSrQs[stage],
                sA=sQ[None, None, None, stage],
                zero_init=True,
            )
            for stage in range(self.q_stage)
        ]
        
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrPs[stage],
                sA=None,
            )
            for stage in range(self.q_stage)
        ]
        
        # ========== Pipeline state ==========
        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)
        
        # ========== Main loop ==========
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        
        while work_tile.is_valid_tile:
            m_block_base, head_idx, batch_idx, split_idx = work_tile.tile_idx
            
            # Adjust for cluster position
            m_block = m_block_base * self.cluster_m + cta_id_in_cluster
            
            seqlen = SeqlenInfoCls(batch_idx)
            
            # Check if this CTA has valid work
            seqlen_q_tiles = cute.ceil_div(seqlen.seqlen_q, self.m_block_size * self.q_stage)
            has_valid_work = (m_block < seqlen_q_tiles)
            
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            block_iter_count = n_block_max - n_block_min
            process_tile = (const_expr(not self.is_split_kv) or n_block_min < n_block_max) and has_valid_work
            
            if process_tile:
                # ========== First QK GEMMs ==========
                for stage in cutlass.range_constexpr(self.q_stage):
                    # Wait for Q
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_load_q_full_offset + stage, 
                        mma_q_consumer_phase
                    )
                    # Wait for K
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    
                    tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self._offset_kv_smem(
                            sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        )
                    
                    # GEMM: S = Q × K^T
                    gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    
                    # Signal S ready
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                
                mma_q_consumer_phase ^= 1
                
                # Release first K
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                
                # ========== Main loop: alternating PV and QK GEMMs ==========
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                
                for i in cutlass.range(block_loop_count, unroll=1):
                    # Wait for V
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Wait for P (from softmax) and rescaled O
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                            P_full_O_rescaled_phase,
                        )
                        
                        # GEMM: O += P × V
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self._offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                            mbar_phase=P_full_O_rescaled_phase,
                        )
                        
                        # Release V, advance to K
                        if const_expr(stage == self.q_stage - 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        
                        # Wait for next K
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        
                        # GEMM: S = Q × K^T (next iteration)
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self._offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        
                        gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                        
                        # Signal S ready
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                    
                    # Release K
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    O_should_accumulate = True
                
                # ========== Release Q ==========
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.q_stage):
                        tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + stage)
                
                # ========== Final PV GEMM ==========
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                
                for stage in cutlass.range_constexpr(self.q_stage):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, 
                        P_full_O_rescaled_phase
                    )
                    
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self._offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=not O_should_accumulate,
                        mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    
                    # Signal O ready
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                
                P_full_O_rescaled_phase ^= 1
                
                # Release final V
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
            
            # ========== Advance to next cluster work unit ==========
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
