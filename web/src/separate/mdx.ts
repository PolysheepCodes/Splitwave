/**
 * MDX-Net separation, ported from python-audio-separator's MDXSeparator.demix / run_model with the
 * settings UtauCover uses (hop 1024, segment 256, overlap 0.25, no denoise).
 */
import { STFT } from "../dsp/stft.ts";
import type { Net, Progress, Stereo } from "./types.ts";

export interface MdxConfig {
  nFft: number;          // 7680 (Voc_FT) / 5120 (KARA_2)
  dimF: number;          // frequency bins the model sees: 3072 / 2048
  compensate: number;    // level correction for the inverted (secondary) stem
  hop?: number;          // 1024
  segment?: number;      // 256 frames per model call
  overlap?: number;      // 0.25
}

/** np.hanning: symmetric Hann of length m. */
function npHanning(m: number): Float32Array {
  const w = new Float32Array(m);
  if (m === 1) { w[0] = 1; return w; }
  for (let i = 0; i < m; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (m - 1));
  return w;
}

export async function mdxDemix(mix: Stereo, net: Net, cfg: MdxConfig, progress?: Progress): Promise<Stereo> {
  const hop = cfg.hop ?? 1024, seg = cfg.segment ?? 256, overlap = cfg.overlap ?? 0.25;
  const { nFft, dimF } = cfg;
  const n = mix[0].length;
  const trim = nFft / 2;
  const chunk = hop * (seg - 1);
  const gen = chunk - 2 * trim;
  const pad = gen + trim - (n % gen);
  const L = trim + n + pad;
  const mixture: Stereo = [new Float32Array(L), new Float32Array(L)];
  mixture[0].set(mix[0], trim);
  mixture[1].set(mix[1], trim);
  const step = Math.floor((1 - overlap) * chunk);
  const result: Stereo = [new Float32Array(L), new Float32Array(L)];
  const divider = new Float32Array(L);
  const stft = new STFT(nFft, hop);
  const T = stft.frames(chunk);                       // = seg
  const part: Stereo = [new Float32Array(chunk), new Float32Array(chunk)];
  const starts: number[] = [];
  for (let start = 0; start < L; start += step) starts.push(start);

  /** STFT of one chunk -> [1, 4, dimF, T], channels (L re, L im, R re, R im), lowest 3 bins zeroed. */
  const prepare = (start: number): Float32Array => {
    const end = Math.min(start + chunk, L);
    const input = new Float32Array(4 * dimF * T);
    for (let c = 0; c < 2; c++) {
      part[c].fill(0);
      part[c].set(mixture[c].subarray(start, end));
      stft.forward(part[c], (t, f, re, im) => {
        if (f < 3) return;
        input[((c * 2) * dimF + f) * T + t] = re;
        input[((c * 2 + 1) * dimF + f) * T + t] = im;
      }, dimF);
    }
    return input;
  };
  const accumulate = (start: number, out: Float32Array) => {
    const end = Math.min(start + chunk, L), actual = end - start;
    const window = overlap !== 0 ? npHanning(actual) : null;
    for (let c = 0; c < 2; c++) {
      const wave = stft.inverse(T, (t, f) => f < dimF
        ? [out[((c * 2) * dimF + f) * T + t], out[((c * 2 + 1) * dimF + f) * T + t]] : [0, 0]);
      const r = result[c];
      for (let i = 0; i < actual; i++) r[start + i] += window ? wave[i] * window[i] : wave[i];
    }
    for (let i = 0; i < actual; i++) divider[start + i] += window ? window[i] : 1;
  };

  // several chunks in flight when the network can take them (CPU worker pools)
  const batch = Math.max(1, net.parallel ?? 1);
  let done = 0;
  for (let b = 0; b < starts.length; b += batch) {
    const group = starts.slice(b, b + batch);
    const outs = await Promise.all(group.map((start) => net.run(prepare(start), [1, 4, dimF, T])));
    group.forEach((start, j) => accumulate(start, outs[j]));
    done += group.length;
    progress?.(done, starts.length);
  }
  const out: Stereo = [new Float32Array(n), new Float32Array(n)];
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < n; i++) {
      const d = divider[trim + i];
      out[c][i] = d > 0 ? result[c][trim + i] / d : 0;
    }
  }
  return out;
}
