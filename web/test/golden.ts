/**
 * Golden test: Splitwave's browser pipeline vs UtauCover's separator (python-audio-separator) on the same
 * 30 s excerpt. Open /test/golden.html?preset=fast|best on the dev server; results land in window.__golden.
 */
import { decodeSong } from "../src/ui/audio.ts";
import type { Stereo } from "../src/separate/types.ts";
import { cpuWorkerCount, spawnCpuWorkers } from "../src/engine/cpu-workers.ts";

const o = document.getElementById("o")!;
const log = (s: string) => { o.textContent += "\n" + s; };
const preset = new URLSearchParams(location.search).get("preset") || "fast";
const backend = new URLSearchParams(location.search).get("backend") || ("gpu" in navigator ? "webgpu" : "wasm");
const result: Record<string, unknown> = { preset, backend };
(window as unknown as { __golden: unknown }).__golden = result;

const load = async (url: string) => decodeSong(await (await fetch(url)).blob());
function snr(ref: Stereo, est: Stereo): number {
  let s = 0, e = 0;
  const n = Math.min(ref[0].length, est[0].length);
  for (let c = 0; c < 2; c++) for (let i = 0; i < n; i++) { s += ref[c][i] ** 2; e += (ref[c][i] - est[c][i]) ** 2; }
  return 10 * Math.log10(s / Math.max(e, 1e-20));
}

try {
  const song = await load("/test/golden/excerpt.wav");
  log(`excerpt: ${(song[0].length / 44100).toFixed(1)} s · preset ${preset} · ${backend}`);
  const worker = new Worker(new URL("../src/engine/worker.ts", import.meta.url), { type: "module" });
  const t0 = performance.now();
  const stems: Record<string, Stereo> = await new Promise((ok, fail) => {
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "done") ok(m.stems);
      else if (m.type === "error") fail(new Error(m.message));
      else if (m.type === "status") log(`  ${m.text}`);
    };
    worker.onerror = (e) => fail(new Error(e.message));
    const cpu = backend === "wasm" ? spawnCpuWorkers(cpuWorkerCount()) : null;
    worker.postMessage({ type: "separate", song, preset, karaoke: true, backend, cpuPorts: cpu?.ports ?? [] }, cpu?.ports ?? []);
  });
  result.seconds = (performance.now() - t0) / 1000;
  log(`separated in ${(result.seconds as number).toFixed(1)} s`);
  for (const name of ["vocals", "instrumental", "lead", "backing"]) {
    const ref = await load(`/test/golden/${preset}/${name}.wav`);
    const db = snr(ref, stems[name]);
    result[name] = db;
    log(`${name.padEnd(13)} vs UtauCover: SNR ${db.toFixed(1)} dB`);
  }
  result.ok = true;
} catch (e) {
  result.error = String(e);
  log("ERROR " + ((e as Error).stack || e));
}
