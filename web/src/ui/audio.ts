/** Decoding songs, writing WAV files, and bundling stems into a ZIP. */
import { SAMPLE_RATE, type Stereo } from "../separate/types.ts";

/** Any format the browser can decode -> stereo at 44.1 kHz (the rate the models use). */
export async function decodeSong(file: Blob): Promise<Stereo> {
  const data = await file.arrayBuffer();
  // decodeAudioData on an OfflineAudioContext resamples to that context's rate
  const ctx = new OfflineAudioContext(2, SAMPLE_RATE, SAMPLE_RATE);
  const buf = await ctx.decodeAudioData(data);
  const left = buf.getChannelData(0).slice();
  const right = buf.numberOfChannels > 1 ? buf.getChannelData(1).slice() : left.slice();
  return [left, right];
}

/** 16-bit PCM WAV, samples scaled by 32767 and truncated, as python-audio-separator writes them. */
export function wav16(s: Stereo, sampleRate = SAMPLE_RATE): Blob {
  const n = s[0].length, bytes = 44 + n * 4;
  const b = new ArrayBuffer(bytes), v = new DataView(b);
  const str = (o: number, t: string) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, bytes - 8, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, n * 4, true);
  const pcm = new Int16Array(b, 44, n * 2);
  const [l, r] = s;
  for (let i = 0; i < n; i++) {
    pcm[2 * i] = Math.trunc(Math.max(-1, Math.min(1, l[i])) * 32767);
    pcm[2 * i + 1] = Math.trunc(Math.max(-1, Math.min(1, r[i])) * 32767);
  }
  return new Blob([b], { type: "audio/wav" });
}

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
function crc32(d: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < d.length; i++) c = CRC_TABLE[(c ^ d[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** A ZIP of the given files, stored uncompressed (WAV doesn't compress much anyway). */
export async function zip(files: { name: string; blob: Blob }[]): Promise<Blob> {
  const enc = new TextEncoder();
  const parts: BlobPart[] = [], central: Uint8Array<ArrayBuffer>[] = [];
  let offset = 0;
  for (const f of files) {
    const data = new Uint8Array(await f.blob.arrayBuffer());
    const name = new Uint8Array(enc.encode(f.name)), crc = crc32(data);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true); local.setUint16(4, 20, true); local.setUint16(8, 0, true);
    local.setUint32(14, crc, true); local.setUint32(18, data.length, true); local.setUint32(22, data.length, true);
    local.setUint16(26, name.length, true);
    parts.push(local.buffer, name, data);
    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true); cd.setUint16(4, 20, true); cd.setUint16(6, 20, true);
    cd.setUint32(16, crc, true); cd.setUint32(20, data.length, true); cd.setUint32(24, data.length, true);
    cd.setUint16(28, name.length, true); cd.setUint32(42, offset, true);
    central.push(new Uint8Array(cd.buffer), name);
    offset += 30 + name.length + data.length;
  }
  const cdSize = central.reduce((a, c) => a + c.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
  end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end.buffer], { type: "application/zip" });
}
