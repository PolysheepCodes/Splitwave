/**
 * Complex FFT for any length (mixed radix 2/3/4/5 with a generic radix for other primes), after KissFFT.
 * The separation models use 2048 (RoFormer), 5120 and 7680 (MDX) point transforms.
 * Data lives in separate real/imaginary Float64Arrays; transforms write to output arrays.
 */
export class FFT {
  readonly n: number;
  private factors: number[] = [];
  private twRe: Float64Array;
  private twIm: Float64Array;
  private scrRe: Float64Array;
  private scrIm: Float64Array;

  constructor(n: number) {
    this.n = n;
    let m = n, p = 4;
    const floorSqrt = Math.floor(Math.sqrt(n));
    while (m > 1) {
      while (m % p) {
        p = p === 4 ? 2 : p === 2 ? 3 : p + 2;
        if (p > floorSqrt) p = m;
      }
      m /= p;
      this.factors.push(p, m);
    }
    this.twRe = new Float64Array(n);
    this.twIm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const a = (-2 * Math.PI * i) / n;
      this.twRe[i] = Math.cos(a);
      this.twIm[i] = Math.sin(a);
    }
    const maxP = Math.max(...this.factors.filter((_, i) => i % 2 === 0));
    this.scrRe = new Float64Array(maxP);
    this.scrIm = new Float64Array(maxP);
  }

  /** out = FFT(in) (forward, unnormalised). in/out must not alias. */
  forward(inRe: Float64Array, inIm: Float64Array, outRe: Float64Array, outIm: Float64Array): void {
    this.work(outRe, outIm, 0, inRe, inIm, 0, 1, 0);
  }

  private work(oRe: Float64Array, oIm: Float64Array, o: number, fRe: Float64Array, fIm: Float64Array, f: number,
               fstride: number, fi: number): void {
    const p = this.factors[fi], m = this.factors[fi + 1];
    const end = o + p * m;
    if (m === 1) {
      for (let k = o, j = f; k < end; k++, j += fstride) { oRe[k] = fRe[j]; oIm[k] = fIm[j]; }
    } else {
      for (let k = o, j = f; k < end; k += m, j += fstride) this.work(oRe, oIm, k, fRe, fIm, j, fstride * p, fi + 2);
    }
    switch (p) {
      case 2: this.bfly2(oRe, oIm, o, fstride, m); break;
      case 3: this.bfly3(oRe, oIm, o, fstride, m); break;
      case 4: this.bfly4(oRe, oIm, o, fstride, m); break;
      case 5: this.bfly5(oRe, oIm, o, fstride, m); break;
      default: this.bflyGeneric(oRe, oIm, o, fstride, m, p);
    }
  }

  private bfly2(re: Float64Array, im: Float64Array, o: number, fs: number, m: number) {
    const tr = this.twRe, ti = this.twIm;
    for (let k = 0; k < m; k++) {
      const a = o + k, b = a + m, w = k * fs;
      const xr = re[b] * tr[w] - im[b] * ti[w], xi = re[b] * ti[w] + im[b] * tr[w];
      re[b] = re[a] - xr; im[b] = im[a] - xi;
      re[a] += xr; im[a] += xi;
    }
  }

  private bfly4(re: Float64Array, im: Float64Array, o: number, fs: number, m: number) {
    const tr = this.twRe, ti = this.twIm;
    for (let k = 0; k < m; k++) {
      const i0 = o + k, i1 = i0 + m, i2 = i1 + m, i3 = i2 + m;
      const w1 = k * fs, w2 = 2 * k * fs, w3 = 3 * k * fs;
      const s0r = re[i1] * tr[w1] - im[i1] * ti[w1], s0i = re[i1] * ti[w1] + im[i1] * tr[w1];
      const s1r = re[i2] * tr[w2] - im[i2] * ti[w2], s1i = re[i2] * ti[w2] + im[i2] * tr[w2];
      const s2r = re[i3] * tr[w3] - im[i3] * ti[w3], s2i = re[i3] * ti[w3] + im[i3] * tr[w3];
      const s5r = re[i0] - s1r, s5i = im[i0] - s1i;
      const r0 = re[i0] + s1r, j0 = im[i0] + s1i;
      const s3r = s0r + s2r, s3i = s0i + s2i;
      const s4r = s0r - s2r, s4i = s0i - s2i;
      re[i2] = r0 - s3r; im[i2] = j0 - s3i;
      re[i0] = r0 + s3r; im[i0] = j0 + s3i;
      // forward transform: (s5 - j*s4), (s5 + j*s4)
      re[i1] = s5r + s4i; im[i1] = s5i - s4r;
      re[i3] = s5r - s4i; im[i3] = s5i + s4r;
    }
  }

  private bfly3(re: Float64Array, im: Float64Array, o: number, fs: number, m: number) {
    const tr = this.twRe, ti = this.twIm;
    const epi3 = ti[fs * m]; // sin(-2*pi/3)
    for (let k = 0; k < m; k++) {
      const i0 = o + k, i1 = i0 + m, i2 = i1 + m, w1 = k * fs, w2 = 2 * k * fs;
      const s1r = re[i1] * tr[w1] - im[i1] * ti[w1], s1i = re[i1] * ti[w1] + im[i1] * tr[w1];
      const s2r = re[i2] * tr[w2] - im[i2] * ti[w2], s2i = re[i2] * ti[w2] + im[i2] * tr[w2];
      const s3r = s1r + s2r, s3i = s1i + s2i;
      const s0r = s1r - s2r, s0i = s1i - s2i;
      const ar = re[i0] - s3r * 0.5, ai = im[i0] - s3i * 0.5;
      const br = s0r * epi3, bi = s0i * epi3;
      re[i0] += s3r; im[i0] += s3i;
      re[i2] = ar + bi; im[i2] = ai - br;
      re[i1] = ar - bi; im[i1] = ai + br;
    }
  }

  private bfly5(re: Float64Array, im: Float64Array, o: number, fs: number, m: number) {
    const tr = this.twRe, ti = this.twIm;
    const yar = tr[fs * m], yai = ti[fs * m], ybr = tr[2 * fs * m], ybi = ti[2 * fs * m];
    for (let u = 0; u < m; u++) {
      const i0 = o + u, i1 = i0 + m, i2 = i1 + m, i3 = i2 + m, i4 = i3 + m;
      const s0r = re[i0], s0i = im[i0];
      const w1 = u * fs, w2 = 2 * u * fs, w3 = 3 * u * fs, w4 = 4 * u * fs;
      const s1r = re[i1] * tr[w1] - im[i1] * ti[w1], s1i = re[i1] * ti[w1] + im[i1] * tr[w1];
      const s2r = re[i2] * tr[w2] - im[i2] * ti[w2], s2i = re[i2] * ti[w2] + im[i2] * tr[w2];
      const s3r = re[i3] * tr[w3] - im[i3] * ti[w3], s3i = re[i3] * ti[w3] + im[i3] * tr[w3];
      const s4r = re[i4] * tr[w4] - im[i4] * ti[w4], s4i = re[i4] * ti[w4] + im[i4] * tr[w4];
      const s7r = s1r + s4r, s7i = s1i + s4i, s10r = s1r - s4r, s10i = s1i - s4i;
      const s8r = s2r + s3r, s8i = s2i + s3i, s9r = s2r - s3r, s9i = s2i - s3i;
      re[i0] = s0r + s7r + s8r; im[i0] = s0i + s7i + s8i;
      const s5r = s0r + s7r * yar + s8r * ybr, s5i = s0i + s7i * yar + s8i * ybr;
      const s6r = s10i * yai + s9i * ybi, s6i = -(s10r * yai) - s9r * ybi;
      re[i1] = s5r - s6r; im[i1] = s5i - s6i;
      re[i4] = s5r + s6r; im[i4] = s5i + s6i;
      const s11r = s0r + s7r * ybr + s8r * yar, s11i = s0i + s7i * ybr + s8i * yar;
      const s12r = -(s10i * ybi) + s9i * yai, s12i = s10r * ybi - s9r * yai;
      re[i2] = s11r + s12r; im[i2] = s11i + s12i;
      re[i3] = s11r - s12r; im[i3] = s11i - s12i;
    }
  }

  private bflyGeneric(re: Float64Array, im: Float64Array, o: number, fs: number, m: number, p: number) {
    const tr = this.twRe, ti = this.twIm, n = this.n, sr = this.scrRe, si = this.scrIm;
    for (let u = 0; u < m; u++) {
      for (let q1 = 0, k = u; q1 < p; q1++, k += m) { sr[q1] = re[o + k]; si[q1] = im[o + k]; }
      for (let q1 = 0, k = u; q1 < p; q1++, k += m) {
        let twidx = 0, accR = sr[0], accI = si[0];
        for (let q = 1; q < p; q++) {
          twidx += fs * k;
          if (twidx >= n) twidx -= n;
          accR += sr[q] * tr[twidx] - si[q] * ti[twidx];
          accI += sr[q] * ti[twidx] + si[q] * tr[twidx];
        }
        re[o + k] = accR; im[o + k] = accI;
      }
    }
  }
}
