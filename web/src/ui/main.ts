/** Splitwave's page: pick a song, choose a quality, watch it separate, listen and download. */
import { decodeSong, wav16, zip } from "./audio.ts";
import { getManifest, isCached, modelHost } from "../engine/models.ts";
import { cpuWorkerCount, spawnCpuWorkers } from "../engine/cpu-workers.ts";
import { PRESETS, stages, type Preset, type StemName } from "../separate/pipeline.ts";
import { SAMPLE_RATE, type Stereo } from "../separate/types.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const STEM_LABEL: Record<StemName, string> = { vocals: "Vocals", instrumental: "Instrumental", lead: "Lead vocals", backing: "Backing vocals" };
const ORDER: StemName[] = ["vocals", "instrumental", "lead", "backing"];

let song: Stereo | null = null;
let songName = "song";
let webgpu = false;
let stems: Partial<Record<StemName, Stereo>> = {};

const fmtTime = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const fmtMB = (b: number) => `${Math.round(b / 1e6)} MB`;

function showError(msg: string | null) {
  $("error").textContent = msg ?? "";
  $("error").classList.toggle("hidden", !msg);
}

// ------------------------------------------------------------------------------------------ setup
async function detectDevice() {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(o?: object): Promise<unknown> } }).gpu;
  // ?cpu forces the CPU backend (for comparing, or when a graphics driver misbehaves)
  const forceCpu = new URLSearchParams(location.search).has("cpu");
  try { webgpu = !forceCpu && !!(gpu && (await gpu.requestAdapter({ powerPreference: "high-performance" }))); } catch { webgpu = false; }
  const note = $("device");
  if (webgpu) {
    note.textContent = "Separation runs on your graphics card (WebGPU).";
  } else {
    note.classList.add("warn");
    note.textContent = forceCpu
      ? "Running on the CPU (?cpu in the address). Only Fast quality is available."
      : "This browser has no WebGPU, so separation runs on the CPU (slower) and only Fast quality is available. " +
        "For Best quality use a current Chrome, Edge or Safari.";
    const best = document.querySelector<HTMLInputElement>('input[value="best"]')!;
    best.disabled = true;
    document.querySelector<HTMLInputElement>('input[value="fast"]')!.checked = true;
  }
  if (!crossOriginIsolated && !webgpu) {
    note.textContent += " (The server isn't sending the headers that let it use more than one CPU core.)";
  }
}

async function describePresets() {
  try {
    const m = await getManifest();
    for (const p of Object.keys(PRESETS) as Preset[]) {
      const ids = [PRESETS[p].vocals, PRESETS[p].karaoke];
      const bytes = ids.reduce((a, id) => a + (m[id]?.bytes ?? 0), 0);
      $(`${p}Size`).textContent = fmtMB(bytes);
      const cached = await Promise.all(ids.map(isCached));
      $(`${p}State`).textContent = cached.every(Boolean) ? "downloaded ✓" : cached.some(Boolean) ? "partly downloaded" : "";
    }
    const host = modelHost() === location.host ? "this site" : modelHost();
    $("modelNote").textContent = `Models download once from ${host} and stay in your browser for next time.`;
  } catch (e) {
    showError(`Couldn't read the model list: ${e instanceof Error ? e.message : e}`);
  }
}

// ------------------------------------------------------------------------------------------ pick a song
async function pickFile(file: File) {
  showError(null);
  $("dropBig").textContent = file.name;
  $("dropSmall").textContent = "Reading…";
  try {
    song = await decodeSong(file);
    songName = file.name.replace(/\.[^.]+$/, "") || "song";
    $("dropSmall").textContent = `${fmtTime(song[0].length / SAMPLE_RATE)} · click to choose another song`;
    $("drop").classList.add("has");
    ($("go") as HTMLButtonElement).disabled = false;
  } catch {
    song = null;
    $("dropBig").textContent = "Drop a song here";
    $("dropSmall").textContent = "or click to choose a file · mp3, wav, flac, m4a, ogg";
    $("drop").classList.remove("has");
    ($("go") as HTMLButtonElement).disabled = true;
    showError(`This browser can't read “${file.name}”. Try an mp3 or wav.`);
  }
}

function wireDrop() {
  const drop = $("drop"), input = $<HTMLInputElement>("file");
  input.addEventListener("change", () => input.files?.[0] && pickFile(input.files[0]));
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); const f = (e as DragEvent).dataTransfer?.files[0]; if (f) pickFile(f); });
  // dropping anywhere on the page works too
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer?.files[0]; if (f && $("work").classList.contains("hidden")) pickFile(f); });
}

// ------------------------------------------------------------------------------------------ separate
function start() {
  if (!song) return;
  showError(null);
  const preset = (document.querySelector<HTMLInputElement>('input[name="preset"]:checked')!.value) as Preset;
  const karaoke = $<HTMLInputElement>("karaoke").checked;
  const plan = stages(preset, karaoke);
  $("pick").classList.add("hidden");
  $("result").classList.add("hidden");
  $("work").classList.remove("hidden");
  $("workTitle").textContent = `Splitting “${songName}”`;
  const list = $("stages");
  list.innerHTML = "";
  const rows = plan.map((s) => {
    const li = document.createElement("li");
    li.className = "wait";
    li.innerHTML = `<span class="label"><span class="what"></span><span class="pct muted mono"></span></span><span class="bar"><i></i></span>`;
    li.querySelector(".what")!.textContent = s.label;
    list.appendChild(li);
    return li;
  });
  const t0 = performance.now();
  const timer = setInterval(() => { $("clock").textContent = fmtTime((performance.now() - t0) / 1000); }, 500);
  const stageStart: number[] = [];

  const worker = new Worker(new URL("../engine/worker.ts", import.meta.url), { type: "module" });
  const cpu = webgpu ? null : spawnCpuWorkers(cpuWorkerCount());
  const copy: Stereo = [song[0].slice(), song[1].slice()];
  worker.postMessage({ type: "separate", song: copy, preset, karaoke, backend: webgpu ? "webgpu" : "wasm", cpuPorts: cpu?.ports ?? [] },
                     [copy[0].buffer, copy[1].buffer, ...(cpu?.ports ?? [])]);
  const finish = () => { clearInterval(timer); worker.terminate(); cpu?.terminate(); $("cancel").onclick = null; };
  $("workNote").textContent = webgpu
    ? "Keep this tab open. Your graphics card does the work, so the computer may get warm and loud."
    : "Keep this tab open. Your processor does the work, so the computer may get warm and loud.";
  $("cancel").onclick = () => {
    finish();
    $("work").classList.add("hidden");
    $("pick").classList.remove("hidden");
  };
  worker.onmessage = (e) => {
    const m = e.data;
    const row = rows[m.stage];
    if (m.type === "status") {
      if (!m.text.startsWith("Loading")) stageStart[m.stage] = performance.now();   // separation starts now
      row.classList.remove("wait");
      row.querySelector(".what")!.textContent = m.text;
      row.querySelector(".pct")!.textContent = "starting…";
      (row.querySelector(".bar > i") as HTMLElement).style.width = "0";
    } else if (m.type === "download") {
      row.querySelector(".pct")!.textContent = `downloading ${fmtMB(m.loaded)} of ${fmtMB(m.total)}`;
      (row.querySelector(".bar > i") as HTMLElement).style.width = `${(100 * m.loaded) / m.total}%`;
    } else if (m.type === "progress") {
      // time per chunk so far in this stage -> time left (chunks can finish several at a time on the CPU)
      const frac = m.done / m.total;
      let eta = "";
      const began = stageStart[m.stage];
      if (began !== undefined && m.done < m.total) {
        const per = (performance.now() - began) / m.done;
        eta = ` · about ${fmtTime(Math.max(1, (per * (m.total - m.done)) / 1000))} left`;
      }
      row.querySelector(".pct")!.textContent = `${Math.round(frac * 100)}%${eta}`;
      (row.querySelector(".bar > i") as HTMLElement).style.width = `${frac * 100}%`;
    } else if (m.type === "done") {
      finish();
      stems = m.stems;
      showResult(m.seconds, preset);
    } else if (m.type === "error") {
      finish();
      $("work").classList.add("hidden");
      $("pick").classList.remove("hidden");
      showError(`Separation failed: ${m.message}`);
    }
  };
  worker.onerror = (e) => {
    finish();
    $("work").classList.add("hidden");
    $("pick").classList.remove("hidden");
    showError(`Separation failed: ${e.message || "the worker stopped"}. If this keeps happening, try Fast quality.`);
  };
}

// ------------------------------------------------------------------------------------------ results
let ctx: AudioContext | null = null;
let buffers: Partial<Record<StemName, AudioBuffer>> = {};
let gains: Partial<Record<StemName, GainNode>> = {};
let sources: AudioBufferSourceNode[] = [];
let playing = false, startedAt = 0, offset = 0, duration = 0;
const muted = new Set<StemName>(), soloed = new Set<StemName>();

function drawWave(canvas: HTMLCanvasElement, s: Stereo) {
  const w = (canvas.width = canvas.clientWidth * devicePixelRatio), h = (canvas.height = 40 * devicePixelRatio);
  const g = canvas.getContext("2d")!;
  g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent");
  const n = s[0].length, per = Math.max(1, Math.floor(n / w));
  for (let x = 0; x < w; x++) {
    let p = 0;
    for (let i = x * per; i < Math.min(n, (x + 1) * per); i += 4) p = Math.max(p, Math.abs(s[0][i]), Math.abs(s[1][i]));
    const bh = Math.max(1, p * h * 0.95);
    g.fillRect(x, (h - bh) / 2, 1, bh);
  }
}

function applyGains() {
  for (const k of ORDER) {
    const g = gains[k];
    if (!g) continue;
    const audible = soloed.size ? soloed.has(k) : !muted.has(k);
    g.gain.value = audible ? 1 : 0;
    document.querySelector(`#stems li[data-stem="${k}"]`)?.classList.toggle("silent", !audible);
  }
}

function stopSources() {
  for (const s of sources) { try { s.stop(); } catch { /* not started */ } }
  sources = [];
}

function play(from: number) {
  if (!ctx) return;
  stopSources();
  for (const k of ORDER) {
    const b = buffers[k];
    if (!b) continue;
    const src = ctx.createBufferSource();
    src.buffer = b;
    src.connect(gains[k]!);
    src.start(0, from);
    sources.push(src);
  }
  startedAt = ctx.currentTime - from;
  playing = true;
  $("play").textContent = "⏸";
}

function pause() {
  if (!ctx) return;
  offset = ctx.currentTime - startedAt;
  stopSources();
  playing = false;
  $("play").textContent = "▶";
}

function tick() {
  if (ctx && playing) {
    const t = ctx.currentTime - startedAt;
    if (t >= duration) { pause(); offset = 0; }
    else $<HTMLInputElement>("seek").value = String(Math.round((t / duration) * 1000));
  }
  const cur = playing && ctx ? ctx.currentTime - startedAt : offset;
  $("time").textContent = `${fmtTime(Math.min(cur, duration))} / ${fmtTime(duration)}`;
  requestAnimationFrame(tick);
}

function showResult(seconds: number, preset: Preset) {
  $("work").classList.add("hidden");
  $("result").classList.remove("hidden");
  ctx?.close();
  ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  buffers = {}; gains = {}; muted.clear(); soloed.clear(); offset = 0; playing = false;
  const list = $("stems");
  list.innerHTML = "";
  for (const k of ORDER) {
    const s = stems[k];
    if (!s) continue;
    const b = ctx.createBuffer(2, s[0].length, SAMPLE_RATE);
    b.copyToChannel(s[0] as Float32Array<ArrayBuffer>, 0);
    b.copyToChannel(s[1] as Float32Array<ArrayBuffer>, 1);
    buffers[k] = b;
    const g = ctx.createGain();
    g.connect(ctx.destination);
    gains[k] = g;
    duration = b.duration;
    const li = document.createElement("li");
    li.dataset.stem = k;
    li.innerHTML = `<span class="name"></span><canvas></canvas><span class="btns">
      <button class="solo" title="Hear only this">Solo</button><button class="mute" title="Silence this">Mute</button>
      <button class="dl" title="Download as WAV">WAV</button></span>`;
    li.querySelector(".name")!.textContent = STEM_LABEL[k];
    list.appendChild(li);
    drawWave(li.querySelector("canvas")!, s);
    li.querySelector(".solo")!.addEventListener("click", (e) => {
      soloed.has(k) ? soloed.delete(k) : soloed.add(k);
      (e.currentTarget as HTMLElement).classList.toggle("on", soloed.has(k));
      applyGains();
    });
    li.querySelector(".mute")!.addEventListener("click", (e) => {
      muted.has(k) ? muted.delete(k) : muted.add(k);
      (e.currentTarget as HTMLElement).classList.toggle("on", muted.has(k));
      applyGains();
    });
    li.querySelector(".dl")!.addEventListener("click", () => download(wav16(s), `${songName} (${STEM_LABEL[k]}).wav`));
  }
  // vocals + instrumental together are the whole song; with lead/backing too, start on those two
  if (stems.lead) { muted.add("vocals"); document.querySelector('#stems li[data-stem="vocals"] .mute')?.classList.add("on"); }
  applyGains();
  $("resultTitle").textContent = songName;
  $("resultInfo").textContent = `${preset === "best" ? "Best" : "Fast"} quality · separated in ${fmtTime(seconds)}`;
}

function download(blob: Blob, name: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

function wireResult() {
  $("play").addEventListener("click", () => (playing ? pause() : play(offset)));
  $<HTMLInputElement>("seek").addEventListener("input", (e) => {
    offset = (Number((e.target as HTMLInputElement).value) / 1000) * duration;
    if (playing) play(offset);
  });
  $("zip").addEventListener("click", async () => {
    const files = ORDER.filter((k) => stems[k]).map((k) => ({ name: `${songName} (${STEM_LABEL[k]}).wav`, blob: wav16(stems[k]!) }));
    download(await zip(files), `${songName} stems.zip`);
  });
  $("again").addEventListener("click", () => {
    pause();
    $("result").classList.add("hidden");
    $("pick").classList.remove("hidden");
    describePresets();
  });
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------------------------------ go
wireDrop();
wireResult();
$("go").addEventListener("click", start);
detectDevice().then(describePresets);
// keep downloaded models from being cleared when the browser is short on space
navigator.storage?.persist?.().catch(() => {});
