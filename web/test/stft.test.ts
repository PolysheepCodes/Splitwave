// node --experimental-strip-types web/test/stft.test.ts  -> compares the TS STFT with torch.stft / torch.istft
import { readFileSync } from "node:fs";
import { STFT } from "../src/dsp/stft.ts";

const dir = new URL("./vectors/", import.meta.url);
const f32 = (name: string) => new Float32Array(readFileSync(new URL(name, dir)).buffer.slice(0));
const x = f32("x.f32");
let failed = false;
for (const [n, hop] of [[2048, 441], [7680, 1024], [5120, 1024]]) {
  const ref = f32(`stft_${n}.f32`);                 // [bins][frames][2]
  const s = new STFT(n, hop), T = s.frames(x.length), bins = s.bins;
  let err = 0, mag = 0;
  const re = new Float32Array(bins * T), im = new Float32Array(bins * T);
  s.forward(x, (t, f, r, i) => {
    re[f * T + t] = r; im[f * T + t] = i;
    const k = (f * T + t) * 2;
    err = Math.max(err, Math.abs(r - ref[k]), Math.abs(i - ref[k + 1]));
    mag = Math.max(mag, Math.abs(ref[k]));
  });
  // inverse of (spectrum * (1 + 0.3j)), same as the reference
  const y = s.inverse(T, (t, f) => {
    const r = re[f * T + t], i = im[f * T + t];
    return [r - 0.3 * i, i + 0.3 * r];
  });
  const yref = f32(`istft_${n}.f32`);
  let ierr = 0;
  for (let k = 0; k < yref.length; k++) ierr = Math.max(ierr, Math.abs(y[k] - yref[k]));
  const ok = err / mag < 1e-5 && ierr < 1e-5 && y.length === yref.length;
  failed ||= !ok;
  console.log(`${ok ? "ok  " : "FAIL"} n_fft ${n}: stft max err ${(err / mag).toExponential(1)} (relative), istft max err ${ierr.toExponential(1)}, length ${y.length}/${yref.length}`);
}
process.exit(failed ? 1 : 0);
