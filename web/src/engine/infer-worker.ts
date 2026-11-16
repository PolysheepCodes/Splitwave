/// <reference lib="webworker" />
/**
 * One single-threaded CPU inference worker. The CPU backend runs several of these side by side, each on a
 * different chunk (onnxruntime's own multithreading deadlocks inside a Web Worker). The page starts them
 * (not every browser lets a worker start workers) and hands each a MessagePort to the separation worker.
 */
import * as ort from "onnxruntime-web/webgpu";

let session: ort.InferenceSession | null = null;
let outputName = "";
let port: MessagePort;
const post = (m: unknown, t: Transferable[] = []) => port.postMessage(m, t);

self.onmessage = (e: MessageEvent) => {
  if (e.data?.type === "port") {
    port = e.data.port;
    port.onmessage = handle;
  }
};

async function handle(e: MessageEvent) {
  const m = e.data;
  try {
    if (m.type === "load") {
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create(m.model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      outputName = m.outputName;
      post({ type: "loaded" });
    } else if (m.type === "run") {
      const out = await session!.run({ [session!.inputNames[0]]: new ort.Tensor("float32", m.input, m.dims) });
      const data = (out[outputName] ?? out[session!.outputNames[0]]).data as Float32Array;
      const copy = data.slice();
      for (const v of Object.values(out)) v.dispose?.();
      post({ type: "result", id: m.id, output: copy }, [copy.buffer]);
    } else if (m.type === "release") {
      await session?.release();
      session = null;
      post({ type: "released" });
    }
  } catch (err) {
    post({ type: "error", id: m.id, message: err instanceof Error ? err.message : String(err) });
  }
}
