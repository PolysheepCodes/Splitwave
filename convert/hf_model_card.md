---
license: other
license_name: per-model
tags:
  - audio-source-separation
  - music-source-separation
  - onnx
  - webgpu
pipeline_tag: audio-to-audio
---

# Splitwave models

Browser-ready ONNX versions of the stem separation models used by
[Splitwave](https://github.com/PolysheepCodes/Splitwave), a web app that splits songs into vocals,
instrumental, lead and backing vocals on the visitor's own computer (WebGPU / WebAssembly).

| File | Model | Does | Precision |
|---|---|---|---|
| `bs_roformer_fp16.onnx` | BS-RoFormer (ep 317, SDR 12.98) by viperx | vocals / instrumental (Best) | fp16 |
| `mel_roformer_karaoke_fp16.onnx` | Mel-Band RoFormer karaoke (SDR 10.20) by aufr33 & viperx | lead / backing vocals (Best) | fp16 |
| `mdx_voc_ft.onnx` | UVR-MDX-NET Voc_FT by the UVR team | vocals / instrumental (Fast) | fp32, unchanged |
| `mdx_kara_2.onnx` | UVR_MDXNET_KARA_2 by the UVR team | lead / backing vocals (Fast) | fp32, unchanged |

`manifest.json` lists each file's size and SHA-256. Splitwave checks downloads against it.

## Conversion

The RoFormer networks were exported from the original PyTorch checkpoints with Splitwave's
[`convert/roformer.py`](https://github.com/PolysheepCodes/Splitwave/blob/main/convert/roformer.py). They take
one 8-second chunk's stereo STFT (`[1, 2050, 801, 2]`, n_fft 2048, hop 441) and return the complex mask to
multiply it with. The STFT and inverse STFT happen in the app. The band split was restructured for WebGPU's
limits, without changing the math. Each export was checked against the original model on real audio
before the fp16 conversion (normalisation layers and rotary position tables kept in fp32).

## Credits and licenses

The weights belong to their authors: the Ultimate Vocal Remover team (MDX-Net models), viperx
(BS-RoFormer), and aufr33 & viperx (Mel-Band RoFormer karaoke). They were obtained through
[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)'s model repository.
This repo redistributes them, converted for in-browser use, under the original authors' terms. Check those
terms before reusing them.
