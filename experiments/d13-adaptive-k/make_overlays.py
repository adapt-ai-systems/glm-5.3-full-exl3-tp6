#!/usr/bin/env python3
"""Build the D13 vLLM overlays from the e3-v2 image copies (docker cp of /usr/local/lib/python3.12/dist-packages/vllm).

  make_overlays.py SRC_VLLM_DIR OUTDIR

- config/compilation.py: GLM_D13_NO_CG_ROUND=1 skips rounding capture sizes up to multiples of num_spec+1,
  so bs*(k+1) sizes for k < num_spec survive.
- v1/cudagraph_dispatcher.py + v1/worker/gpu_model_runner.py: GLM_D13_QLENS="2,3,4,5" adds FULL uniform-decode
  graphs for every listed query length q (keys (bs*q, bs, uniform)), and the runner dispatches a uniform batch
  of any listed q to them. Stock e3-v2 only has FULL decode graphs at q = num_spec+1; every other q runs PIECEWISE.
Unset env = stock behaviour.
"""
import pathlib, sys

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def patch(rel, edits, name):
    s = (src / rel).read_text()
    for old, new in edits:
        assert s.count(old) == 1, (rel, old[:80], s.count(old))
        s = s.replace(old, new)
    (out / name).write_text(s)
    print('wrote', out / name)


patch('config/compilation.py', [
    ("\nimport enum\n", "\nimport enum\nimport os  # [glm53-d13]\n"),
    ("""            and uniform_decode_query_len > 1
        ):
            self.adjust_cudagraph_sizes_for_spec_decode(""",
     """            and uniform_decode_query_len > 1
            and os.environ.get("GLM_D13_NO_CG_ROUND") != "1"  # [glm53-d13] keep k<K capture sizes
        ):
            self.adjust_cudagraph_sizes_for_spec_decode("""),
], 'compilation.py')

patch('v1/cudagraph_dispatcher.py', [
    ("from collections.abc import Set as AbstractSet\n",
     "import os  # [glm53-d13]\nfrom collections.abc import Set as AbstractSet\n"),
    ("""        self.uniform_decode_query_len = 1 + self.vllm_config.num_speculative_tokens
""",
     """        self.uniform_decode_query_len = 1 + self.vllm_config.num_speculative_tokens
        # [glm53-d13] extra uniform-decode query lengths that get FULL decode graphs
        self.decode_qlens = sorted(
            {int(x) for x in os.environ.get("GLM_D13_QLENS", "").split(",") if x.strip()}
            | {self.uniform_decode_query_len}
        )
"""),
    ("""        has_lora: bool,
        num_active_loras: int = 0,
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:""",
     """        has_lora: bool,
        num_active_loras: int = 0,
        uniform_query_len: int | None = None,
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = uniform_query_len or self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

        if (
            uniform_decode
            and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
            and num_tokens_padded % uniform_decode_query_len == 0  # [glm53-d13]
        ):
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
        else:"""),
    ("""            max_num_tokens = (
                uniform_decode_query_len
                * self.vllm_config.scheduler_config.max_num_seqs
            )
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when full mode is enabled."
            )
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs, num_active_loras in product(
                cudagraph_capture_sizes_for_decode, lora_cases
            ):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs, True, num_active_loras > 0, num_active_loras
                    ),
                )
""",
     """            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when full mode is enabled."
            )
            # [glm53-d13] one FULL decode graph set per query length q
            for q in self.decode_qlens:
                max_num_tokens = q * self.vllm_config.scheduler_config.max_num_seqs
                cudagraph_capture_sizes_for_decode = [
                    x
                    for x in self.compilation_config.cudagraph_capture_sizes
                    if x <= max_num_tokens and x >= q and x % q == 0
                ]
                for bs, num_active_loras in product(
                    cudagraph_capture_sizes_for_decode, lora_cases
                ):
                    self.add_cudagraph_key(
                        CUDAGraphMode.FULL,
                        self._create_padded_batch_descriptor(
                            bs, True, num_active_loras > 0, num_active_loras, q
                        ),
                    )
"""),
    ("""        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:""",
     """        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
        uniform_query_len: int | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:"""),
    ("""        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, effective_num_active_loras
        )""",
     """        batch_desc = self._create_padded_batch_descriptor(
            num_tokens,
            normalized_uniform,
            has_lora,
            effective_num_active_loras,
            uniform_query_len,
        )"""),
], 'cudagraph_dispatcher.py')

patch('v1/worker/gpu_model_runner.py', [
    ("""        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,""",
     """        # [glm53-d13] a uniform batch of any query length with FULL decode graphs
        uniform_query_len = (
            max_num_scheduled_tokens
            if max_num_scheduled_tokens in self.cudagraph_dispatcher.decode_qlens
            else self.uniform_decode_query_len
        )
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=uniform_query_len,"""),
    ("""                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
            )""",
     """                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
                uniform_query_len=uniform_query_len,
            )"""),
    ("""        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens
""",
     """        max_query_len = (
            (getattr(self, "_d13_capture_q", None) or self.uniform_decode_query_len)
            if uniform_decode
            else num_tokens
        )  # [glm53-d13]
"""),
    ("""        if profiler is None:
            profiler = nullcontext()
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups""",
     """        if profiler is None:
            profiler = nullcontext()
        # [glm53-d13] capture a uniform desc at its own query length
        self._d13_capture_q = (
            desc.num_tokens // desc.num_reqs if desc.uniform and desc.num_reqs else None
        )
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups"""),
    ('''            torch.accelerator.synchronize()
        self.maybe_remove_all_loras(self.lora_config)
''', '''            torch.accelerator.synchronize()
        self._d13_capture_q = None  # [glm53-d13]
        self.maybe_remove_all_loras(self.lora_config)
'''),
], 'gpu_model_runner.py')
