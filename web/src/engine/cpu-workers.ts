/**
 * Starts the CPU inference workers from the page and returns one MessagePort per worker for the separation
 * worker (see infer-worker.ts).
 */
export function spawnCpuWorkers(count: number): { ports: MessagePort[]; terminate(): void } {
  const workers: Worker[] = [], ports: MessagePort[] = [];
  for (let i = 0; i < count; i++) {
    const w = new Worker(new URL("./infer-worker.ts", import.meta.url), { type: "module" });
    const ch = new MessageChannel();
    w.postMessage({ type: "port", port: ch.port1 }, [ch.port1]);
    workers.push(w);
    ports.push(ch.port2);
  }
  return { ports, terminate: () => workers.forEach((w) => w.terminate()) };
}

/** How many CPU inference workers to run: about half the cores, at most 4 (each holds a model copy). */
export function cpuWorkerCount(): number {
  return Math.max(1, Math.min(4, Math.floor((navigator.hardwareConcurrency || 2) / 2)));
}
