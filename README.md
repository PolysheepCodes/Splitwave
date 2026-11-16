# Splitwave

Split a song into **vocals** and **instrumental**, then the vocals into **lead** and **backing vocals**, in
the browser. The separation runs on the visitor's own graphics card (WebGPU) or processor (WebAssembly).
The server only hands out static files, and songs are never uploaded.

It uses the separation chain tuned for UtauCover, rebuilt for the browser:

| Quality | Vocals / instrumental | Lead / backing | Download (once per visitor) | Needs |
|---|---|---|---|---|
| **Best** | BS-RoFormer (viperx, ep 317) | Mel-Band RoFormer karaoke (aufr33 & viperx) | 786 MB | WebGPU |
| **Fast** | UVR MDX-Net Voc_FT | UVR MDX-Net KARA_2 | 120 MB | any modern browser |

Measured on an M4 MacBook Air (Chrome) for a 3-minute song: Best takes about 9–10 minutes, Fast about
1.5 minutes on WebGPU or about 7 minutes on the CPU. The browser's output matches the original Python
separator (python-audio-separator with UtauCover's settings) to within 41–63 dB per stem. That's far
below anything audible; see [Testing](#testing).

## Running it

```bash
cd web && npm install && npm run build && cd ..
node server/serve.mjs                    # http://localhost:8080 (PORT, APP_DIR, MODELS_DIR to change)
```

The `models/` folder holds the four model files and `manifest.json`. They aren't in git: build them with
[convert/](#the-models) or copy them from another machine.

Development: `cd web && npm run dev` (Vite at http://localhost:5173, serves `../models` at `/models`).

## Deploying on your server

The server's whole job is to serve two folders: `web/dist/` (the app, ~28 MB) at `/`, and `models/` (~900 MB)
at `/models/`. Every page needs these headers. They make the page *cross-origin isolated*, which browsers
require before WebAssembly may use several threads:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

**Node** (no dependencies; put it behind your reverse proxy for HTTPS): `node server/serve.mjs` already
sends the headers, supports range requests, and caches models and hashed assets for a year.

**nginx**:

```nginx
server {
    server_name splitwave.example.com;
    root /srv/splitwave/web/dist;
    add_header Cross-Origin-Opener-Policy same-origin always;
    add_header Cross-Origin-Embedder-Policy require-corp always;
    add_header Cross-Origin-Resource-Policy same-origin always;
    types { application/wasm wasm; }

    location /models/ {
        alias /srv/splitwave/models/;
        add_header Cache-Control "public, max-age=31536000, immutable" always;
        add_header Cross-Origin-Opener-Policy same-origin always;
        add_header Cross-Origin-Embedder-Policy require-corp always;
        add_header Cross-Origin-Resource-Policy same-origin always;
    }
    location /assets/ {
        add_header Cache-Control "public, max-age=31536000, immutable" always;
        add_header Cross-Origin-Opener-Policy same-origin always;
        add_header Cross-Origin-Embedder-Policy require-corp always;
        add_header Cross-Origin-Resource-Policy same-origin always;
    }
    location = /models/manifest.json { alias /srv/splitwave/models/manifest.json; add_header Cache-Control no-cache always; }
}
```

(nginx doesn't inherit `add_header` into a `location` that sets its own, hence the repetition.) HTTPS is
required on anything but `localhost`: browsers only allow WebGPU, cross-origin isolation and the file
storage used for models on secure origins.

**Bandwidth.** Each new visitor downloads the models once: 120 MB for Fast, 786 MB for Best. After that
they're kept in the browser (Origin Private File System) and checked against `manifest.json`. A model only
downloads again if its checksum in the manifest changes.

## How it works

```
web/src/ui          page: decode the song (44.1 kHz), progress, player, WAV / ZIP downloads
web/src/engine      worker.ts runs the chain off the page; net.ts = onnxruntime-web sessions;
                    models.ts = download, checksum, browser storage; infer-worker.ts = CPU worker pool
web/src/separate    pipeline.ts (stem chain and levels), mdx.ts and roformer.ts (chunking and overlap-add)
web/src/dsp         FFT and STFT/iSTFT that match torch.stft / torch.istft
convert/            turns the PyTorch / ONNX models into browser-ready ONNX, writes models/manifest.json
server/serve.mjs    the static server
```

- **Same algorithm as the original.** `mdx.ts` and `roformer.ts` are line-by-line ports of
  python-audio-separator's `demix` with UtauCover's settings (MDX: hop 1024, segment 256, overlap 0.25;
  RoFormer: 8 s chunks every 4 s, Hamming-weighted). `pipeline.ts` reproduces the level handling too: songs
  louder than 0.98 peak are turned down first (otherwise MDX leaves singing in the instrumental), and every
  stem is capped at 0.9 peak.
- **WebGPU when available.** Otherwise the CPU, with up to 4 single-threaded inference workers running
  side by side (onnxruntime's own multithreading deadlocks inside a Web Worker). Add `?cpu` to the address
  to force the CPU.
- **One model in memory at a time**, released between steps.

## The models

`convert/roformer.py` exports the RoFormer models to ONNX that runs on WebGPU:

- The STFT, complex mask multiply and inverse STFT move out of the network into `web/src/dsp`. The network
  maps one chunk's STFT to a complex mask.
- Band splitting is rewritten so no GPU shader binds more than 9 buffers (WebGPU's limit is 10). The
  per-band hidden layers run as one batched matrix multiply. Mel-RoFormer's overlapping bands are averaged
  with two gathers instead of a scatter-add.
- When converting to half precision, the normalisations (sums of squares overflow fp16) and the rotary
  position tables stay in fp32.

Each step is checked against the original model on real audio:

```bash
python convert/roformer.py check  bs --vectors VECTORS   # rewritten network vs the library's model
python convert/roformer.py export bs EXPORT.onnx
python convert/roformer.py validate bs EXPORT.onnx VECTORS
python convert/roformer.py fp16 EXPORT.onnx models/bs_roformer_fp16.onnx
#   ... the same for "mel" -> models/mel_roformer_karaoke_fp16.onnx
python convert/package_models.py                        # copies the MDX models, writes models/manifest.json
```

Run each step in its own process. On a 16 GB machine the export and validation steps peak at 4–7 GB.
Requirements: `pip install -r convert/requirements.txt`. The source checkpoints are the ones
python-audio-separator downloads (`SPLITWAVE_SOURCE_MODELS` points at their folder).

**Licenses.** The model weights belong to their authors (UVR team, viperx, aufr33). Check their terms
before hosting them publicly.

## Testing

- `cd web && npm test`: the TypeScript STFT/iSTFT against torch.stft/istft (1e-7).
- `web/test/golden.html?preset=fast|best[&backend=wasm]` on the dev server: the whole browser pipeline vs
  stems made by UtauCover's separator on the same 30 s excerpt (`web/test/make_reference.py` makes them).
  Results on an M4 Mac:

  | | vocals | instrumental | lead | backing |
  |---|---|---|---|---|
  | Fast (WebGPU and CPU) | 59.1 dB | 63.1 dB | 57.4 dB | 46.7 dB |
  | Best (WebGPU, fp16) | 47.7 dB | 51.3 dB | 43.2 dB* | 40.8 dB* |

  \* Given the same vocals stem as input. In the full chain it's 26.8 / 24.3 dB, because the lead/backing
  model amplifies the tiny differences in the vocals it receives. Two Python runs on different hardware
  differ the same way.
