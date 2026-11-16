/// <reference lib="webworker" />
/**
 * Runs a separation off the page's thread. Messages in: { type: "separate", song, preset, karaoke,
 * backend, cpuPorts } (cpuPorts: MessagePorts to CPU inference workers, for backend "wasm"). Messages out: { type: "status" | "download" | "progress" | "done" | "error", ... }.
 */
import { createCpuPool, createNet, type Backend } from "./net.ts";
import { loadModel } from "./models.ts";
import { separate, stages, type Preset } from "../separate/pipeline.ts";
import type { Stereo } from "../separate/types.ts";

interface SeparateRequest { type: "separate"; song: Stereo; preset: Preset; karaoke: boolean; backend: Backend; cpuPorts?: MessagePort[] }

const post = (msg: unknown, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(msg, transfer);

self.onmessage = async (e: MessageEvent<SeparateRequest>) => {
  const { song, preset, karaoke, backend, cpuPorts = [] } = e.data;
  const plan = stages(preset, karaoke);
  const t0 = performance.now();
  try {
    const stems = await separate(song, preset, karaoke, async (model) => {
      const stage = plan.findIndex((s) => s.model.id === model.id);
      post({ type: "status", stage, text: `Loading ${plan[stage].label.toLowerCase()} model` });
      const bytes = await loadModel(model.id, (loaded, total) => post({ type: "download", stage, loaded, total }));
      post({ type: "status", stage, text: plan[stage].label });
      const output = model.kind === "mdx" ? "output" : "mask";
      return backend === "webgpu" ? createNet(bytes, "webgpu", output) : createCpuPool(bytes, output, cpuPorts);
    }, (stage, done, total) => post({ type: "progress", stage, done, total, elapsed: (performance.now() - t0) / 1000 }));
    const transfer = Object.values(stems).flatMap((s) => s!.map((ch) => ch.buffer));
    post({ type: "done", stems, seconds: (performance.now() - t0) / 1000 }, transfer);
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
