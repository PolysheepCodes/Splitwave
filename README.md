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

## Try it on your computer

You need [Git](https://git-scm.com/downloads) and [Node.js](https://nodejs.org/) 22 or newer.

```bash
git clone https://github.com/PolysheepCodes/Splitwave.git
cd Splitwave/web
npm ci              # installs the build tools into web/node_modules (~250 MB)
npm run build       # builds the app into web/dist
cd ..
node server/serve.mjs
```

Open http://localhost:8080 in Chrome, Edge or Safari. The models download from Hugging Face the first time
you split a song (see [Where the models come from](#where-the-models-come-from)).

For development, `cd web && npm run dev` starts a live-reloading server at http://localhost:5173.

## Installing on a server

These steps are for a Linux server (Ubuntu or Debian) with a domain name pointing at it. Splitwave is a
static site: the server only sends files, and all the separation happens in visitors' browsers. Any small
server works.

**1. Install Git, Node.js and nginx** (Node is only needed to build the app):

```bash
sudo apt update
sudo apt install -y git nginx
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -   # Node.js 22 from NodeSource
sudo apt install -y nodejs
node --version                                                     # v22 or newer
```

**2. Get Splitwave.** It's installed in `/srv/splitwave` here (the config files below assume that path):

```bash
sudo mkdir -p /srv/splitwave
sudo chown "$USER" /srv/splitwave
git clone https://github.com/PolysheepCodes/Splitwave.git /srv/splitwave
```

**3. Build it:**

```bash
cd /srv/splitwave/web
npm ci
npm run build          # -> /srv/splitwave/web/dist, the folder the web server serves
```

**4. Serve it with nginx.** [`server/nginx.conf`](server/nginx.conf) has everything set up: the headers
the app needs, compression and caching. Copy it, put your domain in `server_name`, and enable it:

```bash
sudo cp /srv/splitwave/server/nginx.conf /etc/nginx/sites-available/splitwave
sudo nano /etc/nginx/sites-available/splitwave        # change splitwave.example.com to your domain
sudo ln -s /etc/nginx/sites-available/splitwave /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

**5. Turn on HTTPS.** Browsers only allow WebGPU and the in-browser model storage on secure sites:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your.domain.example
```

Done. Open https://your.domain.example.

**Updating** to the latest version:

```bash
cd /srv/splitwave && git pull
cd web && npm ci && npm run build      # nginx serves the new files right away
```

**Other ways to serve it:**
- **Caddy** instead of nginx gets the HTTPS certificate by itself: [`server/Caddyfile`](server/Caddyfile).
- **The built-in Node server** (`node server/serve.mjs`, port 8080) behind a reverse proxy you already run.
  [`server/splitwave.service`](server/splitwave.service) runs it as a systemd service.
- **Any static host** (Netlify, Cloudflare Pages, a shared host...) works if it can send these two headers
  on every response. They make the page *cross-origin isolated*, which browsers require before WebAssembly
  may use several threads:
  ```
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  ```
- **No Node.js on the server?** Build on your own computer and copy the result:
  `rsync -av web/dist/ you@server:/srv/splitwave/web/dist/`.

## Where the models come from

The four model files (906 MB) and their `manifest.json` (sizes and SHA-256 checksums) are hosted on
Hugging Face. Browsers download them straight from Hugging Face's CDN, so your server never sends them.
Each visitor downloads a model once: 120 MB for Fast, 786 MB for Best. After that it's kept in the browser
(Origin Private File System) and checked against the manifest. A model only downloads again if its
checksum in the manifest changes.

The location is set in [`web/.env`](web/.env) and built into the app:

```
VITE_MODELS_URL=https://huggingface.co/<user>/<repo>/resolve/main/
```

To host the models yourself instead, leave `VITE_MODELS_URL` empty and rebuild. Put the `models/` folder
in `/srv/splitwave/models` and enable the `/models/` block in the nginx config (or set `MODELS_DIR` for
the Node server).

To publish models to your own Hugging Face repo: `hf auth login`, then
`python convert/upload_hf.py <user>/<repo>`. That uploads `models/`, the manifest and a model card.

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
