"""TDD unit tests for ComfyUI core performance & quality fixes (S1, Q4, S3)."""

from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch
from torch import nn

from comfy.cli_args import args

_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    import comfy.ldm.minimax.model as mm
    import comfy.ldm.modules.attention as attn_mod
    import comfy.model_management as model_management
    import comfy.ops as comfy_ops
    import comfy.text_encoders.llama as llama_mod
    import main
finally:
    args.cpu = _original_cpu


class _FakeModel:
    pass


class _FakeModelPatcher:
    def __init__(self, load_device, offload_device, size_bytes=10 * 1024**3, dynamic=False):
        self.load_device = load_device
        self.offload_device = offload_device
        self._size = size_bytes
        self._loaded_size = size_bytes
        self._dynamic = dynamic
        self.parent = None
        self.model = _FakeModel()
        self.partial_unload_calls = 0
        self.detach_calls = 0

    def is_dynamic(self):
        return self._dynamic

    def model_size(self):
        return self._size

    def loaded_size(self):
        return self._loaded_size

    def current_loaded_device(self):
        return self.load_device

    def partially_unload(self, device, memory_to_free):
        self.partial_unload_calls += 1
        if device == self.load_device:
            return 0
        freed = min(self._loaded_size, memory_to_free)
        self._loaded_size -= freed
        return freed

    def detach(self, unpatch_weights=True):
        self.detach_calls += 1
        self._loaded_size = 0


def _make_loaded_model(patcher):
    lm = model_management.LoadedModel(patcher)
    lm.real_model = weakref.ref(patcher.model)
    lm.model_finalizer = weakref.finalize(patcher.model, lambda: None)
    lm.model_finalizer.atexit = False
    return lm


def test_free_memory_skips_same_device_offload_models(monkeypatch):
    cuda0 = torch.device("cuda", 0)
    p1 = _FakeModelPatcher(cuda0, cuda0)
    p2 = _FakeModelPatcher(cuda0, cuda0)
    lm1 = _make_loaded_model(p1)
    lm2 = _make_loaded_model(p2)

    monkeypatch.setattr(model_management, "current_loaded_models", [lm1, lm2])
    monkeypatch.setattr(model_management, "get_free_memory", lambda dev, torch_free_too=False: (1024, 1024) if torch_free_too else 1024)
    monkeypatch.setattr(model_management, "vram_state", model_management.VRAMState.HIGH_VRAM)
    soft_empty = Mock()
    monkeypatch.setattr(model_management, "soft_empty_cache", soft_empty)

    unloaded = model_management.free_memory(5 * 1024**3, cuda0)

    assert unloaded == []
    assert model_management.current_loaded_models == [lm1, lm2]
    assert p1.partial_unload_calls == 0
    assert p1.detach_calls == 0
    assert p2.partial_unload_calls == 0
    assert p2.detach_calls == 0
    assert lm1.currently_used is True
    assert lm2.currently_used is True
    soft_empty.assert_not_called()


def test_free_memory_explicit_unload_all_still_unloads_same_device_models(monkeypatch):
    cuda0 = torch.device("cuda", 0)
    p1 = _FakeModelPatcher(cuda0, cuda0)
    lm1 = _make_loaded_model(p1)

    monkeypatch.setattr(model_management, "current_loaded_models", [lm1])
    monkeypatch.setattr(model_management, "get_free_memory", lambda dev, torch_free_too=False: (1024, 1024) if torch_free_too else 1024)
    monkeypatch.setattr(model_management, "vram_state", model_management.VRAMState.HIGH_VRAM)
    soft_empty = Mock()
    monkeypatch.setattr(model_management, "soft_empty_cache", soft_empty)

    unloaded = model_management.free_memory(1e30, cuda0)

    assert len(unloaded) == 1
    assert model_management.current_loaded_models == []
    assert p1.detach_calls == 1
    soft_empty.assert_called_once()


def test_free_memory_still_unloads_offloadable_models(monkeypatch):
    cuda0 = torch.device("cuda", 0)
    cpu = torch.device("cpu")
    p_resident = _FakeModelPatcher(cuda0, cuda0)
    p_offload = _FakeModelPatcher(cuda0, cpu)
    lm_res = _make_loaded_model(p_resident)
    lm_off = _make_loaded_model(p_offload)

    monkeypatch.setattr(model_management, "current_loaded_models", [lm_res, lm_off])
    monkeypatch.setattr(model_management, "get_free_memory", lambda dev, torch_free_too=False: (1024, 1024) if torch_free_too else 1024)
    monkeypatch.setattr(model_management, "vram_state", model_management.VRAMState.HIGH_VRAM)
    soft_empty = Mock()
    monkeypatch.setattr(model_management, "soft_empty_cache", soft_empty)

    unloaded = model_management.free_memory(20 * 1024**3, cuda0)

    assert len(unloaded) == 1
    assert unloaded[0].model is p_offload
    assert model_management.current_loaded_models == [lm_res]
    assert p_resident.detach_calls == 0
    assert p_offload.detach_calls == 1
    soft_empty.assert_called_once()


class _LoopEscape(Exception):
    pass


class _WorkerQueue:
    def __init__(self, queued_after_first=0, flags=None):
        self.queued = queued_after_first
        self.running = 0
        self.flags = flags or {}
        self.get_calls = 0

    def get(self, timeout=None):
        self.get_calls += 1
        if self.get_calls > 1:
            raise _LoopEscape("second get")
        self.running = 1
        return (0, "pid-1", {}, {}, [], {}), 1

    def get_tasks_remaining(self):
        return self.queued + self.running

    def task_done(self, *args, **kwargs):
        self.running = 0

    def get_flags(self):
        return dict(self.flags)


class _DummyExecutor:
    def __init__(self, *args, **kwargs):
        self.history_result = {}
        self.success = True
        self.status_messages = []

    def execute(self, *args, **kwargs):
        return None

    def reset(self):
        return None


def _run_worker_once(monkeypatch, *, gpu_only, queued_after_first, flags=None):
    monkeypatch.setattr(main.execution, "PromptExecutor", _DummyExecutor)
    monkeypatch.setattr(main.args, "gpu_only", gpu_only)
    clock = iter((1.0, 25.0, 25.0))
    monkeypatch.setattr(main.time, "perf_counter", lambda: next(clock))
    soft_empty = Mock()
    monkeypatch.setattr(main.comfy.model_management, "soft_empty_cache", soft_empty)
    gc_collect = Mock()
    monkeypatch.setattr(main.gc, "collect", gc_collect)
    unload_all = Mock()
    monkeypatch.setattr(main.comfy.model_management, "unload_all_models", unload_all)

    server = SimpleNamespace(last_prompt_id=None, client_id=None)
    asset_manager = SimpleNamespace(
        pause_background_scan=lambda: None,
        resume_background_scan=lambda: None,
        queue_output_scan=lambda: None,
    )
    with pytest.raises(_LoopEscape):
        main.prompt_worker(_WorkerQueue(queued_after_first=queued_after_first, flags=flags), server, asset_manager)
    return soft_empty, gc_collect, unload_all


def test_prompt_worker_skips_soft_empty_cache_under_gpu_only(monkeypatch):
    soft_empty, gc_collect, _ = _run_worker_once(monkeypatch, gpu_only=True, queued_after_first=0)
    soft_empty.assert_not_called()
    gc_collect.assert_called_once()


def test_prompt_worker_defers_gc_while_prompts_remain_queued(monkeypatch):
    soft_empty, gc_collect, _ = _run_worker_once(monkeypatch, gpu_only=True, queued_after_first=1)
    soft_empty.assert_not_called()
    gc_collect.assert_not_called()


def test_prompt_worker_explicit_free_memory_still_calls_soft_empty_cache(monkeypatch):
    soft_empty, gc_collect, unload_all = _run_worker_once(
        monkeypatch, gpu_only=True, queued_after_first=0, flags={"free_memory": True}
    )
    unload_all.assert_called_once()
    gc_collect.assert_called_once()
    soft_empty.assert_called_once()


def test_attention_pytorch_uses_is_causal_only_when_mask_is_none(monkeypatch):
    recorded = []

    def fake_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        recorded.append((attn_mask, is_causal))
        return torch.zeros_like(q)

    monkeypatch.setattr(attn_mod.comfy.ops, "scaled_dot_product_attention", fake_sdpa)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)

    attn_mod.attention_pytorch(q, k, v, 2, mask=None, skip_reshape=True, is_causal=True)
    attn_mod.attention_pytorch(q, k, v, 2, mask=torch.zeros(6, 6), skip_reshape=True, is_causal=True)

    assert recorded[0][0] is None and recorded[0][1] is True
    assert recorded[1][0] is not None and recorded[1][1] is False


def test_llama_uses_native_causal_sdpa_when_no_attention_mask(monkeypatch):
    cfg = llama_mod.Qwen3VL_32BConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    cfg.head_dim = 8
    cfg.rope_dims = [2, 1, 1]
    torch.manual_seed(0)
    model = llama_mod.Llama2_(cfg, device="cpu", dtype=torch.float32, ops=comfy_ops.disable_weight_init).eval()
    for p in model.parameters():
        nn.init.normal_(p, std=0.1)

    seen_calls = []
    orig_attn_pytorch = attn_mod.attention_pytorch

    def spy_attn(q, k, v, heads, mask=None, **kwargs):
        seen_calls.append((mask, kwargs.get("is_causal", False)))
        return orig_attn_pytorch(q, k, v, heads, mask=mask, **kwargs)

    monkeypatch.setattr(model_management, "pytorch_attention_enabled", lambda: True)
    monkeypatch.setattr(llama_mod, "optimized_attention_for_device", lambda *a, **kw: attn_mod.attention_pytorch)

    # First check explicit all-ones attention mask matches native causal run.
    ids = torch.randint(0, 32, (1, 12))
    with torch.no_grad():
        out_explicit, _ = model(ids, attention_mask=torch.ones(1, 12))

    # Now patch attention_pytorch reference inside llama module / attention module to spy.
    monkeypatch.setattr(attn_mod, "attention_pytorch", spy_attn)
    monkeypatch.setattr(llama_mod, "attention_pytorch", spy_attn, raising=False)
    monkeypatch.setattr(llama_mod, "optimized_attention_for_device", lambda *a, **kw: spy_attn)
    with torch.no_grad():
        out_native, _ = model(ids, attention_mask=None)

    assert len(seen_calls) == 2
    assert all(mask is None and is_causal is True for mask, is_causal in seen_calls)
    torch.testing.assert_close(out_native, out_explicit, atol=1e-5, rtol=1e-5)


def test_llama_keeps_explicit_mask_for_padding_and_sliding_window(monkeypatch):
    cfg = llama_mod.Gemma3_4B_Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    cfg.head_dim = 8
    cfg.sliding_attention = [4, False]
    model = llama_mod.Llama2_(cfg, device="cpu", dtype=torch.float32, ops=comfy_ops.disable_weight_init).eval()
    for p in model.parameters():
        nn.init.normal_(p, std=0.1)

    seen_calls = []

    def spy_attn(q, k, v, heads, mask=None, **kwargs):
        seen_calls.append((mask is not None, kwargs.get("is_causal", False)))
        return attn_mod.attention_pytorch(q, k, v, heads, mask=mask, **kwargs)

    monkeypatch.setattr(model_management, "pytorch_attention_enabled", lambda: True)
    monkeypatch.setattr(llama_mod, "optimized_attention_for_device", lambda *a, **kw: spy_attn)
    monkeypatch.setattr(llama_mod, "attention_pytorch", spy_attn, raising=False)

    ids = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        model(ids, attention_mask=None)

    assert seen_calls == [(True, False), (True, False)]


def test_minimax_h3_model_caches_rope_freqs_on_layout(monkeypatch):
    torch.manual_seed(0)
    dm = mm.MiniMaxH3ImageModel(
        pdd_sigmas=[1.0, 0.5, 0.0],
        hidden_size=32,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        ffn_hidden_size=64,
        latents_dim=4,
        text_dim=32,
        time_embed_dim=8,
        rope_inv_freq_len=2,
        adaln_curve_grid=10,
        dtype=torch.float32,
        device="cpu",
        operations=nn,
    ).eval()
    for p in dm.parameters():
        nn.init.normal_(p, std=0.05)
    dm.rope.inv_freq.copy_(torch.tensor([1.0, 0.1]))
    dm.adaln_t_table.uniform_(0.0, 1.0)
    dm.requires_grad_(False)

    calls = 0
    orig_table = mm.rope_rotation_table

    def spy_table(angles, dtype):
        nonlocal calls
        calls += 1
        return orig_table(angles, dtype)

    monkeypatch.setattr(mm, "rope_rotation_table", spy_table)

    lat = torch.randn(1, 4, 1, 4, 4)
    ctx = torch.randn(1, 5, 32)
    layout = mm.PackedLayout(5, 1, 4, 4, 0)
    opts = {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])}

    with torch.no_grad():
        out1 = dm(lat, torch.tensor([1000.0]), ctx, transformer_options=dict(opts), minimax_payload={"layout": layout})
        out2 = dm(lat, torch.tensor([500.0]), ctx, transformer_options=dict(opts), minimax_payload={"layout": layout})
        # Compare step 2 against an uncached fresh layout.
        out2_fresh = dm(
            lat,
            torch.tensor([500.0]),
            ctx,
            transformer_options=dict(opts),
            minimax_payload={"layout": mm.PackedLayout(5, 1, 4, 4, 0)},
        )

    assert calls == 2  # once for cached layout, once for fresh layout (NOT twice for the cached layout!)
    torch.testing.assert_close(out2, out2_fresh)
    assert torch.isfinite(out1).all()


def test_minimax_h3_embed_and_pack_skips_img_update_transfer_when_no_cond():
    dm = mm.MiniMaxH3Model(
        hidden_size=8,
        num_layers=0,
        token_refiner_num_layers=0,
        num_attention_heads=1,
        attention_head_dim=8,
        ffn_hidden_size=8,
        latents_dim=2,
        audio_latents_dim=3,
        text_dim=5,
        timestep_input_dim=8,
        time_embed_hidden_size=8,
        time_embed_dim=8,
        dtype=torch.float32,
        device="cpu",
        operations=nn,
    )
    layout = mm.PackedLayout(4, 1, 4, 6, 2)

    class _ForbidTo:
        def to(self, *args, **kwargs):
            raise AssertionError("img_update/audio_update .to() should not be called when cond rows are absent")

    layout.img_update = _ForbidTo()
    layout.audio_update = _ForbidTo()

    video = torch.randn(1, 2, 1, 4, 6)
    audio = torch.randn(1, 3, 2, 2)
    context = torch.randn(1, 4, 8)

    with torch.no_grad():
        packed = dm._embed_and_pack(video, audio, context, layout, {}, {})
    assert packed.shape == (layout.seq_len, 8)


def test_minimax_dit_block_pre_casts_adaln_matches_reference():
    torch.manual_seed(0)
    block = mm.DiTBlock(
        hidden=16,
        heads=2,
        head_dim=8,
        ffn=32,
        t_dim=6,
        eps=1e-5,
        qk_eps=1e-5,
        apply_silu=False,
        adaln_dtype=torch.float32,
        dtype=torch.bfloat16,
        device="cpu",
        operations=nn,
    ).eval()
    for p in block.parameters():
        nn.init.normal_(p, std=0.1)
    block.requires_grad_(False)

    x = torch.randn(10, 16, dtype=torch.bfloat16)
    t_emb = torch.eye(6, dtype=torch.float32)[:2]
    segs = [(0, 4, 1), (4, 10, 0)]
    rope = torch.randn(1, 10, 1, 2, 2, 2, dtype=torch.bfloat16)

    with torch.no_grad():
        # Reference matching pre-optimization segment-loop semantics
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        h = block.norm1(x)
        for a, b, row in segs:
            h[a:b].mul_(1.0 + scale_msa[row].to(h.dtype)).add_(shift_msa[row].to(h.dtype))
        xr = x.clone()
        attn_out = block.attn(h, rope_freqs=rope, transformer_options={})
        for a, b, row in segs:
            xr[a:b].addcmul_(attn_out[a:b], gate_msa[row].to(xr.dtype))
        h2 = block.norm2(xr)
        for a, b, row in segs:
            h2[a:b].mul_(1.0 + scale_mlp[row].to(h2.dtype)).add_(shift_mlp[row].to(h2.dtype))
        mlp_out = block.mlp(h2)
        for a, b, row in segs:
            xr[a:b].addcmul_(mlp_out[a:b], gate_mlp[row].to(xr.dtype))

        out = block(x.clone(), t_emb, segs, rope, transformer_options={})

    assert torch.equal(out, xr)
