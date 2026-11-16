/**
 * RoFormer separation (BS-RoFormer, Mel-Band-RoFormer), ported from python-audio-separator's
 * MDXCSeparator.demix with UtauCover's settings: 8 s chunks every 4 s (overlap 2), Hamming-weighted
 * overlap-add. The network itself only maps the chunk's STFT to a complex mask (see convert/roformer.py).
 */
import { STFT } from "../dsp/stft.ts";
import type { Net, Progress, Stereo } from "./types.ts";

export interface RoformerConfig {
  nFft?: number;     // 2048
  hop?: number;      // 441
  frames?: number;   // 801 frames = 8 s, fixed in the exported network
  overlap?: number;  // 2
}

/** scipy.signal.windows.hamming: symmetric. */
function hamming(m: number): Float32Array {
  const w = new Float32Array(m);
  for (let i = 0; i < m; i++) w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (m - 1));
  return w;
}

/** Chunk starts that cover the audio without repeating the tail chunk (MDXCSeparator._roformer_chunk_starts). */
export function chunkStarts(length: number, chunk: number, step: number): number[] {
  const starts: number[] = [];
  for (let offset = 0; offset < length; offset += step) {
    if (offset + chunk >= length) {
      const tail = Math.max(length - chunk, 0);
      if (!starts.length || starts[starts.length - 1] !== tail) starts.push(tail);
      break;
    }
    starts.push(offset);
  }
  return starts;
}

export async function roformerDemix(mix: Stereo, net: Net, cfg: RoformerConfig = {}, progress?: Progress): Promise<Stereo> {
  const nFft = cfg.nFft ?? 2048, hop = cfg.hop ?? 441, T = cfg.frames ?? 801, overlap = cfg.overlap ?? 2;
  const chunk = hop * (T - 1);                    // 352800
  const step = Math.floor(chunk / overlap);
  const n = mix[0].length;
  const window = hamming(chunk);
  const result: Stereo = [new Float32Array(n), new Float32Array(n)];
  const counter = new Float32Array(n);
  const stft = new STFT(nFft, hop);
  const F = stft.bins;                             // 1025
  const part = new Float32Array(chunk);
  const starts = chunkStarts(n, chunk, step);

  /** STFT of one chunk -> [1, 2050, T, 2]: (bin, channel) rows, then (re, im). */
  const prepare = (start: number): Float32Array => {
    const length = Math.min(chunk, n - start);
    const spec = new Float32Array(2 * F * T * 2);
    for (let c = 0; c < 2; c++) {
      part.fill(0);                                 // a song shorter than 8 s is zero-padded to the chunk
      part.set(mix[c].subarray(start, start + length));
      stft.forward(part, (t, f, re, im) => {
        const k = (((f * 2 + c) * T) + t) * 2;
        spec[k] = re; spec[k + 1] = im;
      });
    }
    return spec;
  };
  const accumulate = (start: number, spec: Float32Array, mask: Float32Array) => {
    const length = Math.min(chunk, n - start);
    for (let c = 0; c < 2; c++) {
      const wave = stft.inverse(T, (t, f) => {
        const k = (((f * 2 + c) * T) + t) * 2;
        const xr = spec[k], xi = spec[k + 1], mr = mask[k], mi = mask[k + 1];
        return [xr * mr - xi * mi, xr * mi + xi * mr];
      }, chunk);
      const r = result[c];
      for (let i = 0; i < length; i++) r[start + i] += wave[i] * window[i];
    }
    for (let i = 0; i < length; i++) counter[start + i] += window[i];
  };

  const batch = Math.max(1, net.parallel ?? 1);
  let done = 0;
  for (let b = 0; b < starts.length; b += batch) {
    const group = starts.slice(b, b + batch);
    const specs = group.map(prepare);
    const masks = await Promise.all(specs.map((spec) => net.run(spec, [1, 2 * F, T, 2])));
    group.forEach((start, j) => accumulate(start, specs[j], masks[j]));
    done += group.length;
    progress?.(done, starts.length);
  }
  for (let c = 0; c < 2; c++) {
    const r = result[c];
    for (let i = 0; i < n; i++) r[i] /= Math.max(counter[i], 1e-10);
  }
  return result;
}
