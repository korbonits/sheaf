# ESMFold2 kit benchmark — biohub/ESMFold2

- GPU: NVIDIA H100 80GB HBM3 (cc 9.0, 81079 MiB); nvidia-smi: ['NVIDIA H100 80GB HBM3, 580.95.05, 81559 MiB, 1980 MHz, 700.00 W']
- torch 2.13.0+cu130 / CUDA 13.0 / cuDNN 92000 / triton 3.7.1
- flash-attn 2.8.3.post1 / Transformer Engine 2.15.0+42b8400 / Python 3.12.10
- esm 3.3.0 (https://github.com/Biohub/esm.git@26b0bc2b771e3e419ea74f445a5f35cc094a1509)
- transformers 4.57.6 (https://github.com/huggingface/transformers.git@ef32577f55da19a4989cd7b22e004dc43a4998cb)
- esmfold2_opt 0.3.27; kits https://github.com/anthropics/uplifting-biomolecular-modeling.git@f4f62fa6592ae4938d49b1757bea0cfeff9f468e
- sheaf None @ 0c4b40eba4769b0ba702f7cca527809dd1ea2ac3
- measured 2026-09-26T04:48:54+00:00; num_loops=20, num_sampling_steps=200, one sample; 3 timed folds per length (1 warm-up)

Load time: off 32.6 s, exact 51.3 s, fast 76.7 s

`off` = the library's fused backend with chunking off, the configuration `exact` reproduces bit for bit. Time = `backend.predict` wall (one fold + response serialisation). Speedup = off p50 / mode p50.

| L | mode | predict p50 s | p95 s | speedup | peak alloc MiB |
|---:|---|---:|---:|---:|---:|
| 128 | off | 2.83 | 2.84 | 1.00x | 13,617 |
| 128 | exact | 1.11 | 1.22 | 2.56x | 14,621 |
| 128 | fast | 0.36 | 0.61 | 7.76x | 15,597 |
| 256 | off | 3.48 | 3.48 | 1.00x | 14,378 |
| 256 | exact | 2.15 | 2.24 | 1.62x | 15,898 |
| 256 | fast | 0.92 | 1.00 | 3.77x | 16,828 |
| 512 | off | 8.75 | 8.78 | 1.00x | 17,653 |
| 512 | exact | 5.33 | 5.54 | 1.64x | 20,227 |
| 512 | fast | 2.96 | 3.06 | 2.96x | 20,820 |
| 768 | off | 18.88 | 18.88 | 1.00x | 23,109 |
| 768 | exact | 11.03 | 11.10 | 1.71x | 26,798 |
| 768 | fast | 6.46 | 6.56 | 2.92x | 25,520 |

Kit lines (exact):

    [esmfold2-opt] WEIGHTS pinned sha256: 14 file(s) for variant full_nomsa (config.json=c5566fab6a17, model-00001-of-00006.safetensors=bd90149ff223, model-00002-of-00006.safetensors=f75e2144d826, model-00003-of-00006.safetensors=f699f01ecc96, +10 more); 14 (cached digest 2026-09-25T06:01:18Z)
    [esmfold2-opt] ACTIVE mode=exact variant=full_nomsa server_mode=opt7x esm=3.3.0 transformers=4.57.6 gpu=NVIDIA H100 80GB HBM3(sm90) n_gpu=1 sharding=none levers=fused,t6,t3,t5,ax,fz,tg,sg,eg,ec,fc,pb,ls,xtr,xte,trimul,glue,disto,xln,ro fallbacks=none applied=deferred atom_attn=flash_attn esmc_mlp=te esmc_attn=sdpa(chain_mask) esmc_rope=flash_attn_triton flash_attn=2.8.3.post1 transformer_engine=2.15.0+42b8400 xformers=0.0.35+03b91d7.d20260904 compile=n/a graph_lru_sampler=2 graph_budgets=800/800/1536
    [esmfold2-opt] APPLIED model#0 msa_encoder mode=exact variant=full_nomsa server_mode=opt7x levers=fused,t6,t3,t5,ax,fz,tg,sg,eg,ec,fc,pb,ls,xtr,xte,trimul,glue,disto,xln,ro fallbacks=none substituted=none not_for_variant=rg,mh atom_attn=flash_attn atom_forward=instance:_swa_forward_cachedx9 esmc_mlp=te esmc_attn=sdpa(chain_mask) esmc_rope=flash_attn_triton flash_attn=2.8.3.post1 transformer_engine=2.15.0+42b8400 xformers=0.0.35+03b91d7.d20260904
    [esmfold2-opt] LEVER name=fused state=on impl=ef2_server.py origin=kit strategy=F4.autocast_policy mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=tg state=on impl=ef2_opt.py origin=kit strategy=F3.cuda_graph_trunk mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=sg state=on impl=ef2_opt.py origin=kit strategy=F3.cuda_graph_sampler mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=eg state=on impl=ef2_opt.py origin=kit strategy=F3.cuda_graph_trunk mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=ec state=on impl=ef2_opt.py origin=kit strategy=F6.feature_cache mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=fc state=on impl=ef2_opt.py origin=kit strategy=F6.feature_cache mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=pb state=on impl=ef2_opt.py origin=kit strategy=LOCAL.step_invariant_hoist mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=msa state=off impl=ef2_opt.py origin=kit strategy=LOCAL.esmfold2.msa_vendor_fused_trimul mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t1 state=off impl=ef2_w4.py origin=kit strategy=LOCAL.fused_transition mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t3 state=on impl=ef2_w4.py origin=kit strategy=LOCAL.esmfold2.tile_table mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t5 state=on impl=ef2_w4.py origin=kit strategy=F4.bf16_weight_precast mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t6 state=on impl=ef2_w4.py origin=kit strategy=LOCAL.fused_transition mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t10 state=off impl=ef2_w4.py origin=kit strategy=LOCAL.fused_transition mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=mk state=off impl=ef2_mk_sampler.py origin=kit strategy=LOCAL.step_invariant_hoist mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t11 state=off impl=ef2_msa.py origin=kit strategy=F5.fpf_msa_kernels mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t12 state=off impl=ef2_msa.py origin=kit strategy=F5.fpf_msa_kernels mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t13 state=off impl=ef2_msa.py origin=kit strategy=F5.fpf_msa_kernels mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t14 state=off impl=ef2_msa.py origin=kit strategy=LOCAL.layernorm_kernel mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=ls state=on impl=ef2_opt.py origin=kit strategy=LOCAL.esmfold2.loop_static_io mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=rg state=skipped reason=not_for_variant:full_nomsa impl=ef2_opt.py origin=kit strategy=F3.cuda_graph_trunk mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=ax state=on impl=ef2_atom.py origin=kit strategy=LOCAL.step_invariant_hoist mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=af state=off impl=ef2_atom.py origin=kit strategy=LOCAL.esmfold2.atom_block_fused mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=fz state=on impl=ef2_feats.py origin=kit strategy=LOCAL.esmfold2.msa_features_vectorised mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=m15 state=off impl=ef2_msa_v2.py origin=kit strategy=F5.fpf_msa_kernels mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=m16 state=off impl=ef2_msa_v2.py origin=kit strategy=F5.fpf_msa_kernels mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=m17 state=off impl=ef2_msa_v2.py origin=kit strategy=F5.fpf_msa_kernels mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=mh state=skipped reason=not_for_variant:full_nomsa impl=ef2_msa_v2.py origin=kit strategy=LOCAL.step_invariant_hoist mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t15 state=off impl=ef2_pair_v2.py origin=kit strategy=LOCAL.fused_transition mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t15msa state=off impl=ef2_pair_v2.py origin=kit strategy=LOCAL.fused_transition mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=xtr state=on impl=ef2_pair_v2.py origin=kit strategy=LOCAL.fused_transition mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 word=exact provider=opt_core.kernels.transition core=0.5.228.0
    [esmfold2-opt] LEVER name=xte state=on impl=ef2_xte.py origin=kit strategy=LOCAL.fused_transition mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 word=exact provider=opt_core.kernels.transition core=0.5.228.0 modules=58
    [esmfold2-opt] LEVER name=t16 state=off impl=ef2_transition_cute.py origin=kit strategy=LOCAL.fused_transition mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=tx state=off impl=ef2_w4.py origin=kit strategy=F2.fpf_trimul_fast mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=trimul state=on impl=ef2_hoist.py origin=kit strategy=LOCAL.esmfold2.msa_trimul_replumb mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=glue state=on impl=ef2_hoist.py origin=kit strategy=LOCAL.esmfold2.msa_trimul_replumb mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 sigmoid_check=ok:65282
    [esmfold2-opt] LEVER name=disto state=on impl=ef2_hoist.py origin=kit strategy=F6.output_overlap mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=xln state=on impl=ef2_xln.py origin=kit strategy=F5.row_layernorm mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 sites=340 floor_rows=4096 core=0.5.228.0 kernels=6 cc=9.0 provider=opt_core.kernels.ln.exactln
    [esmfold2-opt] LEVER name=ro state=on impl=ef2_dit.py origin=kit strategy=F3.cuda_graph_sampler mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 scope=nds1_batch1 kabsch=torch
    [esmfold2-opt] LEVER name=kd state=off impl=ef2_dit.py origin=kit strategy=F6.host_sync_elimination mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=dit state=off impl=ef2_dit.py origin=kit strategy=LOCAL.dit_fused_kernels mode=exact variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=x2b state=off impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=F7.chunked_eval mode=exact variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x3 state=off impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=F7.chunked_eval mode=exact variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x6 state=off impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=F7.chunked_eval mode=exact variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x7 state=off impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=F7.chunked_eval mode=exact variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x8 state=off impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=LOCAL.esmfold2.lean_init mode=exact variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x10 state=off impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=LOCAL.esmfold2.disto_to_host mode=exact variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x4 state=off impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=LOCAL.esmfold2.lm_host_offload mode=exact variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] NOTE pair: the transition provider refuses word=exact for xte (rows=5776, c=256): exact_vouch_below_400_tokens_on_H100:torch2.13.0+cu130/3.7.1 -> the module's own statement serves these calls (by name)
    [esmfold2-opt] NOTE pair: word=exact resolves to the statement (torch_swiglu) for xtr at this size / class -> the module's own statement serves (by name)
    [esmfold2-opt] NOTE pair: the transition provider refuses word=exact for xtr (rows=76, c=128): exact_vouch_below_20_tokens_on_H100:torch2.13.0+cu130/3.7.1 -> the module's own statement serves these calls (by name)
    [esmfold2-opt] NOTE pair: the transition provider refuses word=exact for xtr (rows=5776, c=256): flash_sm90a:init_during_capture -> the module's own statement serves these calls (by name)

Kit lines (fast):

    [esmfold2-opt] WEIGHTS pinned sha256: 14 file(s) for variant full_nomsa (config.json=c5566fab6a17, model-00001-of-00006.safetensors=bd90149ff223, model-00002-of-00006.safetensors=f75e2144d826, model-00003-of-00006.safetensors=f699f01ecc96, +10 more); 14 (cached digest 2026-09-25T06:01:18Z)
    [esmfold2-opt] ACTIVE mode=fast variant=full_nomsa server_mode=opt14_msa esm=3.3.0 transformers=4.57.6 gpu=NVIDIA H100 80GB HBM3(sm90) n_gpu=1 sharding=none levers=fused,t3,t5,tx,ax,af,fz,tg,sg,eg,ec,fc,pb,msa,ls,m15,m16,m17,t16,xln,ro,kd,dit,x2b,x3,x6,x7,x8,x10 fallbacks=none applied=deferred atom_attn=flash_attn esmc_mlp=te esmc_attn=sdpa(chain_mask) esmc_rope=flash_attn_triton flash_attn=2.8.3.post1 transformer_engine=2.15.0+42b8400 xformers=0.0.35+03b91d7.d20260904 compile=n/a graph_lru_sampler=2 graph_budgets=800/800/1536
    [esmfold2-opt] APPLIED model#0 msa_encoder mode=fast variant=full_nomsa server_mode=opt14_msa levers=fused,t3,t5,tx,ax,af,fz,tg,sg,eg,ec,fc,pb,msa,ls,m15,m16,m17,t16,xln,ro,kd,dit,x2b,x3,x6,x7,x8,x10 fallbacks=none substituted=none not_for_variant=rg,mh not_for_class=t10,t15,t15msa atom_attn=flash_attn atom_forward=instance:_swa_forward_cachedx9 esmc_mlp=te esmc_attn=sdpa(chain_mask) esmc_rope=flash_attn_triton flash_attn=2.8.3.post1 transformer_engine=2.15.0+42b8400 xformers=0.0.35+03b91d7.d20260904
    [esmfold2-opt] LEVER name=fused state=on impl=ef2_server.py origin=kit strategy=F4.autocast_policy mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=tg state=on impl=ef2_opt.py origin=kit strategy=F3.cuda_graph_trunk mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=sg state=on impl=ef2_opt.py origin=kit strategy=F3.cuda_graph_sampler mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=eg state=on impl=ef2_opt.py origin=kit strategy=F3.cuda_graph_trunk mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=ec state=on impl=ef2_opt.py origin=kit strategy=F6.feature_cache mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=fc state=on impl=ef2_opt.py origin=kit strategy=F6.feature_cache mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=pb state=on impl=ef2_opt.py origin=kit strategy=LOCAL.step_invariant_hoist mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=msa state=on impl=ef2_opt.py origin=kit strategy=LOCAL.esmfold2.msa_vendor_fused_trimul mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t1 state=off impl=ef2_w4.py origin=kit strategy=LOCAL.fused_transition mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t3 state=on impl=ef2_w4.py origin=kit strategy=LOCAL.esmfold2.tile_table mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t5 state=on impl=ef2_w4.py origin=kit strategy=F4.bf16_weight_precast mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t6 state=off impl=ef2_w4.py origin=kit strategy=LOCAL.fused_transition mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t10 state=skipped reason=not_for_class:sm90:superseded_by_t16 impl=ef2_w4.py origin=kit strategy=LOCAL.fused_transition mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=mk state=off impl=ef2_mk_sampler.py origin=kit strategy=LOCAL.step_invariant_hoist mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t11 state=off impl=ef2_msa.py origin=kit strategy=F5.fpf_msa_kernels mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t12 state=off impl=ef2_msa.py origin=kit strategy=F5.fpf_msa_kernels mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t13 state=off impl=ef2_msa.py origin=kit strategy=F5.fpf_msa_kernels mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t14 state=off impl=ef2_msa.py origin=kit strategy=LOCAL.layernorm_kernel mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=ls state=on impl=ef2_opt.py origin=kit strategy=LOCAL.esmfold2.loop_static_io mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=rg state=skipped reason=not_for_variant:full_nomsa impl=ef2_opt.py origin=kit strategy=F3.cuda_graph_trunk mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=ax state=on impl=ef2_atom.py origin=kit strategy=LOCAL.step_invariant_hoist mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=af state=on impl=ef2_atom.py origin=kit strategy=LOCAL.esmfold2.atom_block_fused mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 gemm=bf16
    [esmfold2-opt] LEVER name=fz state=on impl=ef2_feats.py origin=kit strategy=LOCAL.esmfold2.msa_features_vectorised mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=m15 state=on impl=ef2_msa_v2.py origin=kit strategy=F5.fpf_msa_kernels mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=m16 state=on impl=ef2_msa_v2.py origin=kit strategy=F5.fpf_msa_kernels mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=m17 state=on impl=ef2_msa_v2.py origin=kit strategy=F5.fpf_msa_kernels mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=mh state=skipped reason=not_for_variant:full_nomsa impl=ef2_msa_v2.py origin=kit strategy=LOCAL.step_invariant_hoist mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t15 state=skipped reason=not_for_class:sm90:superseded_by_t16 impl=ef2_pair_v2.py origin=kit strategy=LOCAL.fused_transition mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t15msa state=skipped reason=not_for_class:sm90:superseded_by_t16 impl=ef2_pair_v2.py origin=kit strategy=LOCAL.fused_transition mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=xtr state=off impl=ef2_pair_v2.py origin=kit strategy=LOCAL.fused_transition mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=xte state=off impl=ef2_xte.py origin=kit strategy=LOCAL.fused_transition mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=t16 state=on impl=ef2_transition_cute.py origin=kit strategy=LOCAL.fused_transition mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 word=fast provider=opt_core.kernels.transition core=0.5.228.0 family=esmpair+esmfused
    [esmfold2-opt] LEVER name=tx state=on impl=ef2_w4.py origin=kit strategy=F2.fpf_trimul_fast mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 word=fast row=tx_sm90a cell=9.0|bf16|C256|H256|N<=256|in|fwd stack=H100:2.13.0+cu130/3.7.1/nocueq abi=torch2.13.0+cu130-cpython-312-x86_64-linux-gnu-sm90 cc=9.0 refused=none stock_classes=0
    [esmfold2-opt] LEVER name=trimul state=off impl=ef2_hoist.py origin=kit strategy=LOCAL.esmfold2.msa_trimul_replumb mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=glue state=off impl=ef2_hoist.py origin=kit strategy=LOCAL.esmfold2.msa_trimul_replumb mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=disto state=off reason=dropped_by_line:fast impl=ef2_hoist.py origin=kit strategy=F6.output_overlap mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90
    [esmfold2-opt] LEVER name=xln state=on impl=ef2_xln.py origin=kit strategy=F5.row_layernorm mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 sites=340 floor_rows=4096 core=0.5.228.0 kernels=6 cc=9.0 provider=opt_core.kernels.ln.exactln
    [esmfold2-opt] LEVER name=ro state=on impl=ef2_dit.py origin=kit strategy=F3.cuda_graph_sampler mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 scope=nds1_batch1 kabsch=device
    [esmfold2-opt] LEVER name=kd state=on impl=ef2_dit.py origin=kit strategy=F6.host_sync_elimination mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 scope=ndsN_batchN
    [esmfold2-opt] LEVER name=dit state=on impl=ef2_dit.py origin=kit strategy=LOCAL.dit_fused_kernels mode=fast variant=full_nomsa model=0 tier_vs_stock=T2 sm=sm90 scope=ndsN_batchN gemm=bf16 cond=bf16 attn=flash attn_precision=bf16
    [esmfold2-opt] LEVER name=x2b state=on impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=F7.chunked_eval mode=fast variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x3 state=on impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=F7.chunked_eval mode=fast variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x6 state=on impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=F7.chunked_eval mode=fast variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x7 state=on impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=F7.chunked_eval mode=fast variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x8 state=on impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=LOCAL.esmfold2.lean_init mode=fast variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x10 state=on impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=LOCAL.esmfold2.disto_to_host mode=fast variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
    [esmfold2-opt] LEVER name=x4 state=off impl=../EF2_XL_ADDON_v1/ef2_xl.py origin=kit strategy=LOCAL.esmfold2.lm_host_offload mode=fast variant=full_nomsa model=0 tier_vs_stock=bitwise sm=sm90
