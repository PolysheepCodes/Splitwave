/**
 * STFT / inverse STFT that reproduce torch.stft / torch.istft (center=True, reflect padding, periodic
 * Hann window, onesided, not normalised), which is what the separation models were trained and run with.
 */
import { FFT } from "./fft.ts";

export function hann(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n); // periodic
  return w;
}

export class STFT {
  readonly nFft: number;
  readonly hop: number;
  readonly bins: number;
  readonly window: Float64Array;
  private fft: FFT;
  private aRe: Float64Array; private aIm: Float64Array; private bRe: Float64Array; private bIm: Float64Array;

  constructor(nFft: number, hop: number) {
    this.nFft = nFft;
    this.hop = hop;
    this.bins = nFft / 2 + 1;
    this.window = hann(nFft);
    this.fft = new FFT(nFft);
    this.aRe = new Float64Array(nFft); this.aIm = new Float64Array(nFft);
    this.bRe = new Float64Array(nFft); this.bIm = new Float64Array(nFft);
  }

  frames(length: number): number {
    return Math.floor(length / this.hop) + 1;
  }

  /**
   * One channel -> spectrum. `put(frame, bin, re, im)` receives each value, so callers write straight into
   * whatever tensor layout their model wants.
   */
  forward(x: Float32Array, put: (t: number, f: number, re: number, im: number) => void, maxBin = this.bins): number {
    const n = this.nFft, pad = n / 2, len = x.length, T = this.frames(len), w = this.window;
    const { aRe, aIm, bRe, bIm } = this;
    for (let t = 0; t < T; t++) {
      const start = t * this.hop - pad;
      for (let i = 0; i < n; i++) {
        let j = start + i;
        if (j < 0) j = -j;                       // reflect (torch pad_mode="reflect")
        else if (j >= len) j = 2 * (len - 1) - j;
        aRe[i] = x[j] * w[i];
        aIm[i] = 0;
      }
      this.fft.forward(aRe, aIm, bRe, bIm);
      for (let f = 0; f < maxBin; f++) put(t, f, bRe[f], bIm[f]);
    }
    return T;
  }

  /**
   * Spectrum -> one channel. `get(frame, bin)` returns [re, im] for bins < `bins` (higher bins are zero).
   * Output length: `length`, or hop * (frames - 1) like torch.istft without a length.
   */
  inverse(T: number, get: (t: number, f: number) => [number, number], length?: number): Float32Array {
    const n = this.nFft, hop = this.hop, w = this.window, bins = this.bins;
    const total = n + hop * (T - 1);
    const y = new Float64Array(total), env = new Float64Array(total);
    const { aRe, aIm, bRe, bIm } = this;
    for (let t = 0; t < T; t++) {
      // inverse via the forward transform: ifft(X) = conj(fft(conj(X))) / n, with the Hermitian half filled in
      for (let f = 0; f < bins; f++) {
        let [re, im] = get(t, f);
        if (f === 0 || f === bins - 1) im = 0;     // irfft ignores the imaginary part at DC and Nyquist
        aRe[f] = re; aIm[f] = -im;
        if (f > 0 && f < bins - 1) { aRe[n - f] = re; aIm[n - f] = im; }
      }
      this.fft.forward(aRe, aIm, bRe, bIm);
      const o = t * hop;
      for (let i = 0; i < n; i++) {
        y[o + i] += (bRe[i] / n) * w[i];
        env[o + i] += w[i] * w[i];
      }
    }
    const start = n / 2;
    const outLen = length ?? total - n;
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const k = start + i;
      out[i] = k < total && env[k] > 1e-11 ? y[k] / env[k] : 0;
    }
    return out;
  }
}
