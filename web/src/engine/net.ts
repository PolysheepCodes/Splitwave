/** onnxruntime-web sessions: WebGPU when the browser has it, otherwise WebAssembly on the CPU. */
import * as ort from "onnxruntime-web/webgpu";
import type { Net } from "../separate/types.ts";

export type Backend = "webgpu" | "wasm";

let configured = false;
function configure(backend: Backend) {
  if (configured) return;
  configured = true;
  // (the WebAssembly file is bundled by Vite next to this code; onnxruntime finds it on its own)
  // Inside a Web Worker, onnxruntime's WebAssembly build deadlocks while starting its own thread pool, so it
  // gets one thread there (enough for WebGPU, where the graphics card does the work). The CPU backend runs
  // on the page instead (see cpu-bridge.ts), where several threads work; they need a cross-origin isolated
  // page (the server's COOP/COEP headers).
  const inWorker = typeof (self as { WorkerGlobalScope?: unknown }).WorkerGlobalScope !== "undefined";
  ort.env.wasm.numThreads = inWorker || backend === "webgpu" || !self.crossOriginIsolated
    ? 1 : Math.min(navigator.hardwareConcurrency || 4, 16);
}

export async function hasWebGPU(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(o?: object): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try { return !!(await gpu.requestAdapter({ powerPreference: "high-performance" })); } catch { return false; }
}

export interface LoadedNet extends Net {
  backend: Backend;
  release(): Promise<void>;
}

/**
 * CPU backend: single-threaded inference workers (started by the page, reached through MessagePorts), each
 * with its own copy of the model. `run` calls go to whichever is free, so the demix loops can keep several
 * chunks in flight (Net.parallel).
 */
export async function createCpuPool(model: Uint8Array, outputName: string, ports: MessagePort[]): Promise<LoadedNet> {
  if (!ports.length) throw new Error("no CPU workers were started");
  type Pending = { ok: (d: Float32Array) => void; fail: (e: Error) => void };
  const pending = new Map<number, Pending>();
  const idle: MessagePort[] = [];
  const waiting: ((p: MessagePort) => void)[] = [];
  let nextId = 0;
  await Promise.all(ports.map((port) => new Promise<void>((ok, fail) => {
    port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "loaded") ok();
      else if (m.type === "result") { pending.get(m.id)?.ok(m.output); pending.delete(m.id); }
      else if (m.type === "error") {
        const err = new Error(m.message);
        if (m.id === undefined) fail(err); else { pending.get(m.id)?.fail(err); pending.delete(m.id); }
      }
    };
    port.postMessage({ type: "load", model: model.slice(), outputName });
  })));
  idle.push(...ports);
  const acquire = () => new Promise<MessagePort>((ok) => { const p = idle.pop(); if (p) ok(p); else waiting.push(ok); });
  const free = (p: MessagePort) => { const next = waiting.shift(); if (next) next(p); else idle.push(p); };
  return {
    backend: "wasm",
    parallel: ports.length,
    async run(input: Float32Array, dims: number[]) {
      const port = await acquire();
      const id = nextId++;
      try {
        return await new Promise<Float32Array>((ok, fail) => {
          pending.set(id, { ok, fail });
          const copy = input.slice();
          port.postMessage({ type: "run", id, input: copy, dims }, [copy.buffer]);
        });
      } finally {
        free(port);
      }
    },
    async release() {
      for (const port of ports) port.postMessage({ type: "release" });
    },
  };
}

export async function createNet(model: Uint8Array, backend: Backend, outputName: string): Promise<LoadedNet> {
  configure(backend);
  const session = await ort.InferenceSession.create(model, {
    executionProviders: [backend],
    graphOptimizationLevel: "all",
  });
  const inputName = session.inputNames[0];
  return {
    backend,
    async run(input: Float32Array, dims: number[]) {
      const feeds = { [inputName]: new ort.Tensor("float32", input, dims) };
      const out = await session.run(feeds);
      const t = out[outputName] ?? out[session.outputNames[0]];
      const data = (t.data as Float32Array).slice();   // copy out before the tensor is disposed
      for (const v of Object.values(out)) v.dispose?.();
      return data;
    },
    async release() { await session.release(); },
  };
}
