# ========== Continuing FlashAttentionForwardSm100Cluster4 class ==========
    
    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        """
        Epilogue warp: write output from SMEM to global memory.
        
        This is similar to the base kernel. Each CTA writes its own output
        to the correct location based on cluster-adjusted M-block.
        """
        epi_consumer_phase = Int32(0)
        
        # Get cluster position for M-block adjustment
        cta_id_in_cluster = cute.arch.cluster_cta_id()
        
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
            
            if (const_expr(not self.is_split_kv) or n_block_min < n_block_max) and has_valid_work:
                # Setup output tensor
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
                gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
                
                if const_expr(self.use_tma_O):
                    # TMA store path
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Wait for correction to finish
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, 
                            epi_consumer_phase
                        )
                        
                        # TMA store
                        store_O(src_idx=stage, dst_idx=self.q_stage * m_block + stage)
                        cute.arch.cp_async_bulk_commit_group()
                    
                    # Wait for stores to complete
                    for stage in cutlass.range_constexpr(self.q_stage):
                        if const_expr(self.q_stage == 2):
                            cute.arch.cp_async_bulk_wait_group(1 - stage, read=True)
                        else:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)
                else:
                    # Manual copy path
                    tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
                    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                    tOsO = gmem_thr_copy_O.partition_S(sO)
                    cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                    tOgO = gmem_thr_copy_O.partition_D(gO)
                    tOcO = gmem_thr_copy_O.partition_S(cO)
                    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                    tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                    
                    pack_gqa = PackGQA(
                        self.m_block_size,
                        self.head_dim_v_padded,
                        self.check_hdim_v_oob,
                        self.qhead_per_kvhead,
                    )
                    
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Wait for correction to finish
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, 
                            epi_consumer_phase
                        )
                        
                        # Load from SMEM to registers
                        tOrO = cute.make_fragment_like(tOsO[None, None, None, 0], self.o_dtype)
                        cute.autovec_copy(tOsO[None, None, None, stage], tOrO)
                        
                        # Store to global memory
                        if const_expr(not self.pack_gqa):
                            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                                if (
                                    t0OcO[0, rest_m, 0][0]
                                    < seqlen.seqlen_q
                                    - (self.q_stage * m_block + stage) * self.m_block_size
                                    - tOcO[0][0]
                                ):
                                    cute.copy(
                                        gmem_tiled_copy_O,
                                        tOrO[None, rest_m, None],
                                        tOgO[None, rest_m, None, self.q_stage * m_block + stage],
                                        pred=tOpO[None, rest_m, None] if const_expr(self.check_hdim_v_oob) else None,
                                    )
                        else:
                            pack_gqa.store_O(
                                mO_cur, tOrO, gmem_tiled_copy_O, tidx,
                                self.q_stage * m_block + stage, seqlen.seqlen_q,
                            )
                        
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)
                
                epi_consumer_phase ^= 1
            
            # Advance to next cluster work unit
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
