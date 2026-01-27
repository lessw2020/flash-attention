# ========== Continuing FlashAttentionForwardSm100Cluster4 class ==========
    
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d)
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d)
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv)
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors = None,
        aux_tensors: Optional[list] = None,
    ):
        """
        Execute Flash Attention with cluster-4 multicast optimization.
        
        This method sets up tensors, TMA descriptors, and launches the kernel.
        """
        # ========== Setup dtypes ==========
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        
        # ========== Assume aligned strides ==========
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mQ, mK, mV, mO)
        ]
        
        # ========== Transpose layouts for kernel ==========
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        
        if const_expr(self.is_split_kv):
            O_layout_transpose = [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            num_splits = Int32(1)
        
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        
        # V needs different transpose for MN-major access
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))
        
        # ========== Validate layouts ==========
        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        
        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of mV is not supported")
        
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        
        # ========== Setup derived attributes ==========
        self._setup_attributes()
        self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None and mSeqUsedQ is None
        
        # e2e frequency tuning
        self.e2e_freq = 16
        if const_expr(
            self.head_dim_padded > 64 and not self.is_causal and not self.is_local and self.pack_gqa
        ):
            self.e2e_freq = 32 if mCuSeqlensQ is not None or mSeqUsedQ is not None else 10
        
        # ========== Create MMA operations ==========
        cta_group = tcgen05.CtaGroup.ONE
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )
        
        # ========== Cluster layout ==========
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )
        
        self.epi_tile = self.mma_tiler_pv[:2]
        
        # ========== Create SMEM layouts ==========
        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage,
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.acc_stage,
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage,
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage,
        )
        
        # Adjust strides for different K/V head dimensions
        if const_expr(not self.same_hdim_kv_padded):
            stride_sK = const_expr(max(sK_layout.outer.stride[-1], 0))
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner, 0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner, 0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )
        
        # ========== Apply PackGQA if enabled ==========
        if const_expr(self.pack_gqa):
            mQ, mO, mLSE = self._apply_pack_gqa(mQ, mO, mLSE, mK)
        
        # ========== Compute TMA copy sizes ==========
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        
        # ========== Create TMA atoms ==========
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        
        # Q TMA (no multicast - each CTA loads different Q)
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk, tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )
        
        # K and V TMA with cluster support
        if const_expr(self.use_tma_KV):
            # For cluster multicast, TMA atoms are created with cluster shape
            # The actual multicast happens in the load function
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op, mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk, tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op, mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv, tiled_mma_pv,
                self.cluster_layout_vmnk.shape,
            )
        else:
            tma_atom_K = None
            tma_atom_V = None
        
        # O TMA (store)
        o_cta_v_layout = cute.composition(cute.make_identity_layout(mO.shape), self.epi_tile)
        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op, mO,
                cute.select(sO_layout, mode=[0, 1]),
                o_cta_v_layout,
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)
        
        # ========== Setup tile scheduler ==========
        # For cluster, we need cluster-aware scheduling
        TileScheduler = self._select_tile_scheduler(mCuSeqlensQ, mSeqUsedQ)
        
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]) if const_expr(mCuSeqlensQ is None) else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0]) if const_expr(mPageTable is None) else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[0],
            total_q=cute.size(mQ.shape[0]) if const_expr(mCuSeqlensQ is not None) else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        
        # ========== Setup barriers ==========
        self._setup_barrier_offsets()
        
        # ========== Create shared storage struct ==========
        sO_size, sQ_size, sK_size = self._compute_smem_sizes(sQ_layout, sK_layout, sO_layout)
        
        @cute.struct
        class SharedStorage:
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            tmem_holding_buf: Int32
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sK_size],
                self.buffer_align_bytes,
            ]
        
        self.shared_storage = SharedStorage
        
        # ========== Compute softmax scale ==========
        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softmax_scale_mod = None
        else:
            softmax_scale_log2 = LOG2_E
            softmax_scale_mod = softmax_scale
        
        # ========== Window sizes ==========
        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)
        
        # ========== Setup divmod helpers ==========
        fastdiv_mods = None
        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = (
                cute.size(mK.shape[0])
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            )
            fastdiv_mods = (FastDivmodDivisor(seqlen_q), FastDivmodDivisor(seqlen_k))
        
        head_divmod = None
        if cutlass.const_expr(self.pack_gqa):
            head_divmod = FastDivmodDivisor(self.qhead_per_kvhead)
        
        # ========== Multicast mask for cluster ==========
        multicast_mask = self._compute_multicast_mask() if self.use_multicast_kv else 1
        
        # ========== Launch kernel ==========
        self.kernel(
            mQ, mK, mV, mO, mLSE,
            mCuSeqlensQ, mCuSeqlensK,
            mSeqUsedQ, mSeqUsedK,
            mPageTable,
            tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O,
            softmax_scale_log2, softmax_scale_mod,
            window_size_left, window_size_right,
            learnable_sink,
            blocksparse_tensors,
            sQ_layout, sK_layout, tP_layout, sV_layout, sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk, tiled_mma_pv,
            tile_sched_params,
            num_splits,
            multicast_mask,  # NEW: pass multicast mask
            aux_tensors,
            fastdiv_mods,
            head_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,  # Cluster launch!
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
    
    def _apply_pack_gqa(self, mQ, mO, mLSE, mK):
        """Apply PackGQA tensor transformation."""
        shape_Q_packed = (
            (self.qhead_per_kvhead, mQ.shape[0]),
            mQ.shape[1],
            mK.shape[2],
            *mQ.shape[3:],
        )
        stride_Q_packed = (
            (mQ.stride[2], mQ.stride[0]),
            mQ.stride[1],
            mQ.stride[2] * self.qhead_per_kvhead,
            *mQ.stride[3:],
        )
        mQ = cute.make_tensor(
            mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
        )
        
        shape_O_packed = (
            (self.qhead_per_kvhead, mO.shape[0]),
            mO.shape[1],
            mK.shape[2],
            *mO.shape[3:],
        )
        stride_O_packed = (
            (mO.stride[2], mO.stride[0]),
            mO.stride[1],
            mO.stride[2] * self.qhead_per_kvhead,
            *mO.stride[3:],
        )
        mO = cute.make_tensor(
            mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
        )
        
        if const_expr(mLSE is not None):
            shape_LSE_packed = (
                (self.qhead_per_kvhead, mLSE.shape[0]),
                mK.shape[2],
                *mLSE.shape[2:],
            )
            stride_LSE_packed = (
                (mLSE.stride[1], mLSE.stride[0]),
                mLSE.stride[1] * self.qhead_per_kvhead,
                *mLSE.stride[2:],
            )
            mLSE = cute.make_tensor(
                mLSE.iterator, cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed)
            )
        
        return mQ, mO, mLSE
    
    def _select_tile_scheduler(self, mCuSeqlensQ, mSeqUsedQ):
        """Select appropriate tile scheduler based on configuration."""
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            return SingleTileVarlenScheduler
        else:
            if const_expr(self.is_causal or self.is_local):
                return SingleTileLPTScheduler
            else:
                return (
                    SingleTileScheduler
                    if const_expr(not self.is_persistent)
                    else StaticPersistentTileScheduler
                )
