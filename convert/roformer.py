"""Convert the RoFormer separation models (BS-RoFormer, Mel-Band-RoFormer) to ONNX that runs in a browser.

The browser does the STFT, the complex multiply and the inverse STFT itself (see web/src/dsp). The exported
network takes the stereo STFT of one chunk and returns the complex mask to multiply it with:

    input  "stft"  float32 [1, 2050, T, 2]   bins x {left, right} interleaved, then (real, imag); T = 801
    output "mask"  float32 [1, 2050, T, 2]   same layout

Rewritten for WebGPU, with the same weights and the same math:
- The original splits the spectrum into 60-62 bands with one Split and joins them with one Concat. WebGPU
  lets a shader bind at most ~10 buffers, so bands are sliced one by one and joined in groups of 8.
- The mask estimators' hidden layers have the same shape for every band, so they run as one batched
  matrix multiply instead of 60 small ones (faster on a GPU too).
- Mel-RoFormer's overlapping bands are averaged back onto frequencies with two Gathers (each frequency is
  in at most two bands) instead of a scatter-add.

    python convert/roformer.py check  bs   # rewritten network vs the original model on real audio
    python convert/roformer.py export bs out.onnx
    python convert/roformer.py validate bs out.onnx VECTORS_DIR   # exported network vs PyTorch
    python convert/roformer.py fp16  in.onnx out.onnx

Run each step in its own process: exporting holds the model, its trace and the serialized file at once.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch import nn

MODELS = Path(os.environ.get("SPLITWAVE_SOURCE_MODELS",
                             Path.home() / "Library/Application Support/UtauCover/models"))
SOURCES = {
    "bs": ("model_bs_roformer_ep_317_sdr_12.9755.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.yaml"),
    "mel": ("mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml"),
}
N_FFT, HOP = 2048, 441
CHUNK = 352800                      # 8 s at 44.1 kHz, the models' training chunk
FRAMES = CHUNK // HOP + 1           # 801
FP32_OPS = ["ReduceL2", "ReduceSumSquare", "ReduceMean", "Pow", "Sqrt", "Clip", "Div"]
FAN_IN = 8                          # inputs per Concat (WebGPU: <= 10 buffers per shader, incl. the output)


def load(kind: str):
    ckpt, conf = SOURCES[kind]
    cfg = yaml.load(open(MODELS / conf), Loader=yaml.UnsafeLoader)
    if kind == "bs":
        from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer as Model
    else:
        from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer as Model
    params = inspect.signature(Model.__init__).parameters
    model = Model(**{k: v for k, v in cfg["model"].items() if k in params})
    sd = torch.load(MODELS / ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    del sd
    gc.collect()
    return model.eval().requires_grad_(False), cfg


def tree_cat(xs: list[torch.Tensor], dim: int) -> torch.Tensor:
    while len(xs) > 1:
        xs = [torch.cat(xs[i:i + FAN_IN], dim=dim) if len(xs[i:i + FAN_IN]) > 1 else xs[i]
              for i in range(0, len(xs), FAN_IN)]
    return xs[0]


class WebCore(nn.Module):
    def __init__(self, model, kind: str):
        super().__init__()
        self.kind = kind
        self.layers = model.layers
        self.final_norm = getattr(model, "final_norm", None) if kind == "bs" else None
        dims = list(model.band_split.dim_inputs)
        self.offsets = np.concatenate([[0], np.cumsum(dims)]).tolist()
        self.band_in = model.band_split.to_features            # per band: RMSNorm(dim_in), Linear(dim_in, dim)
        if len(model.mask_estimators) != 1:
            raise ValueError("one stem per model is supported")
        mlps = [seq[0] for seq in model.mask_estimators[0].to_freqs]  # per band: MLP (then GLU)
        linears = [[m for m in mlp if isinstance(m, nn.Linear)] for mlp in mlps]
        depth = len(linears[0])
        # hidden layers: identical shapes across bands -> stacked [bands, in, out]
        self.hidden_w = nn.ParameterList([nn.Parameter(torch.stack([l[i].weight.t() for l in linears]), False)
                                          for i in range(depth - 1)])
        self.hidden_b = nn.ParameterList([nn.Parameter(torch.stack([l[i].bias for l in linears])[:, None, :], False)
                                          for i in range(depth - 1)])
        self.last = nn.ModuleList([l[-1] for l in linears])    # per band: Linear(hidden, 2 * dim_in)
        if kind == "mel":
            fi = model.freq_indices.long()                      # gathered entry -> (freq, channel) row
            self.register_buffer("freq_indices", fi, persistent=False)
            rows = 2 * (N_FFT // 2 + 1)
            first = torch.full((rows,), -1, dtype=torch.long)
            second = torch.full((rows,), -1, dtype=torch.long)
            for j, r in enumerate(fi.tolist()):
                if first[r] < 0:
                    first[r] = j
                elif second[r] < 0:
                    second[r] = j
                else:
                    raise ValueError("a frequency is in more than two bands")
            if (first < 0).any():
                raise ValueError("a frequency is in no band")
            n = len(fi)                                         # index n = an all-zero row appended below
            second = torch.where(second < 0, torch.full_like(second, n), second)
            count = (second < n).float() + 1.0
            self.register_buffer("src_a", first, persistent=False)
            self.register_buffer("src_b", second, persistent=False)
            self.register_buffer("inv_count", (1.0 / count)[None, :, None, None], persistent=False)

    def forward(self, stft: torch.Tensor) -> torch.Tensor:     # [1, 2050, T, 2]
        x = stft.index_select(1, self.freq_indices) if self.kind == "mel" else stft
        x = rearrange(x, "b f t c -> b t (f c)")
        bands = []
        for k, net in enumerate(self.band_in):
            xb = x[..., self.offsets[k]:self.offsets[k + 1]]
            bands.append(net(xb).unsqueeze(2))
        x = tree_cat(bands, dim=2)                              # [b, t, bands, dim]
        for block in self.layers:
            time_t, freq_t = block[-2], block[-1]
            b, t, f, d = x.shape
            x = time_t(x.permute(0, 2, 1, 3).reshape(b * f, t, d))
            x = freq_t(x.reshape(b, f, t, d).permute(0, 2, 1, 3).reshape(b * t, f, d)).reshape(b, t, f, d)
        if self.final_norm is not None:
            x = self.final_norm(x)
        b, t, k, d = x.shape
        h = x.permute(2, 0, 1, 3).reshape(k, b * t, d)        # bands first: one batched matmul per layer
        for w, bias in zip(self.hidden_w, self.hidden_b):
            h = torch.tanh(torch.matmul(h, w) + bias)
        outs = [F.glu(lin(h[i]), dim=-1) for i, lin in enumerate(self.last)]
        mask = tree_cat(outs, dim=-1).reshape(b, t, -1)
        mask = rearrange(mask, "b t (f c) -> b f t c", c=2)
        if self.kind == "mel":                                  # average overlapping bands per frequency
            mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)
            mask = (mask.index_select(1, self.src_a) + mask.index_select(1, self.src_b)) * self.inv_count
        return mask


# ------------------------------------------------------------------------------------------- checks
def stft(audio: torch.Tensor) -> torch.Tensor:
    """[2, n] -> [1, 2050, T, 2], the layout the network takes (what web/src/dsp computes)."""
    st = torch.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=torch.hann_window(N_FFT),
                    return_complex=True)
    return rearrange(torch.view_as_real(st).unsqueeze(0), "b s f t c -> b (f s) t c").contiguous()


def apply(st: torch.Tensor, mask: torch.Tensor, length: int) -> torch.Tensor:
    y = torch.view_as_complex(st) * torch.view_as_complex(mask.contiguous())
    y = rearrange(y, "b (f s) t -> (b s) f t", s=2)
    return torch.istft(y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=torch.hann_window(N_FFT),
                       length=length)


def test_audio() -> torch.Tensor:
    import librosa
    song = os.environ.get("SPLITWAVE_TEST_SONG", str(Path.home() / "Downloads/Here Come The Cats.mp3"))
    y, _ = librosa.load(song, sr=44100, mono=False, offset=60, duration=CHUNK / 44100)
    y = y[:, :CHUNK] / max(1.0, float(np.abs(y).max()) / 0.98)
    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))


def snr_db(ref: torch.Tensor, est: torch.Tensor) -> float:
    return float(10 * torch.log10((ref ** 2).sum() / ((ref - est) ** 2).sum().clamp(min=1e-20)))


def cmd_check(kind: str, vectors: str | None):
    torch.set_grad_enabled(False)
    model, _ = load(kind)
    audio = test_audio()
    original = model(audio.unsqueeze(0))[0]                 # the library's own forward: STFT -> net -> iSTFT
    core = WebCore(model, kind).eval()
    st = stft(audio)
    mask = core(st)
    mine = apply(st, mask, audio.shape[-1])
    n = min(original.shape[-1], mine.shape[-1])
    print(f"{kind}: rewritten network vs original model on 8 s of a song: SNR {snr_db(original[:, :n], mine[:, :n]):.1f} dB, "
          f"max abs diff {float((original[:, :n] - mine[:, :n]).abs().max()):.2e}", flush=True)
    if vectors:
        out = Path(vectors)
        out.mkdir(parents=True, exist_ok=True)
        st.numpy().tofile(out / f"{kind}_stft_in.f32")
        mask.numpy().tofile(out / f"{kind}_mask_ref.f32")
        print(f"wrote test vectors to {out}")


def split_wide_nodes(model, fan_in: int = FAN_IN) -> int:
    """Rewrite every Concat with more than fan_in inputs as a tree of Concats. (The exporter's graph passes
    flatten nested torch.cat calls back into one, so the grouping in WebCore doesn't survive export.)"""
    import onnx
    from onnx import helper
    nodes, changed, counter = [], 0, 0
    for n in model.graph.node:
        if n.op_type != "Concat" or len(n.input) <= fan_in:
            nodes.append(n)
            continue
        axis = next(a.i for a in n.attribute if a.name == "axis")
        level = list(n.input)
        while len(level) > fan_in:
            nxt = []
            for i in range(0, len(level), fan_in):
                part = level[i:i + fan_in]
                if len(part) == 1:
                    nxt.append(part[0])
                    continue
                counter += 1
                out = f"{n.output[0]}__part{counter}"
                nodes.append(helper.make_node("Concat", part, [out], axis=axis, name=f"{n.name}__part{counter}"))
                nxt.append(out)
            level = nxt
        nodes.append(helper.make_node("Concat", level, list(n.output), axis=axis, name=n.name))
        changed += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model) if model.ByteSize() < 2**31 else None
    return changed


def cmd_export(kind: str, dst: str):
    torch.set_grad_enabled(False)
    model, _ = load(kind)
    core = WebCore(model, kind).eval()
    x = torch.randn(1, 2 * (N_FFT // 2 + 1), FRAMES, 2) * 0.1
    # warm up first: the rotary embeddings build their position table on first use, and tracing that
    # first-use path exported a Split along the wrong axis in layer 0
    core(x)
    torch.onnx.export(core, (x,), dst, input_names=["stft"], output_names=["mask"], opset_version=17,
                      dynamo=False, do_constant_folding=True)
    import onnx
    m = onnx.load(dst)
    split = split_wide_nodes(m)
    onnx.save(m, dst)
    worst = max((len(n.input) + len(n.output), n.op_type) for n in m.graph.node)
    print(f"split {split} wide Concat node(s) into trees")
    print(f"exported {dst}: {os.path.getsize(dst) / 1e6:.0f} MB, {len(m.graph.node)} nodes, "
          f"most buffers in one node: {worst[0]} ({worst[1]})")


def cmd_validate(kind: str, src: str, vectors: str):
    """Run the exported fp32 network in onnxruntime (CPU) on the test vectors from `check --vectors`."""
    import onnxruntime as ort
    x = np.fromfile(Path(vectors) / f"{kind}_stft_in.f32", dtype=np.float32).reshape(1, -1, FRAMES, 2)
    ref = np.fromfile(Path(vectors) / f"{kind}_mask_ref.f32", dtype=np.float32).reshape(x.shape)
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False   # keeps the peak down (~8 GB with the arena on a 16 GB Mac)
    so.intra_op_num_threads = 4
    got = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"]).run(None, {"stft": x})[0]
    snr = 10 * np.log10((ref ** 2).sum() / max(((ref - got) ** 2).sum(), 1e-20))
    print(f"{kind}: onnxruntime vs PyTorch mask: SNR {snr:.1f} dB, max abs diff {np.abs(ref - got).max():.2e}")


def position_nodes(model) -> list[str]:
    """Nodes that compute the rotary sin/cos tables: Sin/Cos and everything feeding them that doesn't
    depend on the audio (positions only)."""
    prod = {o: n for n in model.graph.node for o in n.output}
    audio = set(i.name for i in model.graph.input)
    depends: dict[str, bool] = {}

    def from_audio(name: str) -> bool:
        if name in audio:
            return True
        if name not in prod:
            return False
        if name not in depends:
            depends[name] = False
            depends[name] = any(from_audio(i) for i in prod[name].input if i)
        return depends[name]

    picked, stack = set(), [n for n in model.graph.node if n.op_type in ("Sin", "Cos")]
    while stack:
        n = stack.pop()
        if n.name in picked:
            continue
        picked.add(n.name)
        stack += [prod[i] for i in n.input if i in prod and not from_audio(i)]
    return sorted(picked)


def cmd_fp16(src: str, dst: str):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m = onnx.load(src)
    # Normalisation sums squares of raw spectrum values, which overflow half precision (max 65504) and
    # turn the whole output into NaN: those ops stay in float32 (casts are inserted around them).
    # The rotary embeddings' angles (position x frequency) reach ~800 rad, where half precision only
    # resolves steps of 0.5: the small subgraph that computes them stays in float32 too.
    m16 = convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=False,
                                   op_block_list=FP32_OPS, node_block_list=position_nodes(m))
    del m
    onnx.save(m16, dst)
    print(f"fp16 {dst}: {os.path.getsize(dst) / 1e6:.0f} MB")


def main():
    sys.setrecursionlimit(100_000)   # graph walks over ~6000-node models
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("kind", choices=SOURCES)
    c.add_argument("--vectors", help="also write the network's input / reference output here")
    e = sub.add_parser("export")
    e.add_argument("kind", choices=SOURCES)
    e.add_argument("dst")
    v = sub.add_parser("validate")
    v.add_argument("kind", choices=SOURCES)
    v.add_argument("src")
    v.add_argument("vectors")
    h = sub.add_parser("fp16")
    h.add_argument("src")
    h.add_argument("dst")
    a = ap.parse_args()
    if a.cmd == "check":
        cmd_check(a.kind, a.vectors)
    elif a.cmd == "export":
        cmd_export(a.kind, a.dst)
    elif a.cmd == "validate":
        cmd_validate(a.kind, a.src, a.vectors)
    else:
        cmd_fp16(a.src, a.dst)


if __name__ == "__main__":
    sys.exit(main())
