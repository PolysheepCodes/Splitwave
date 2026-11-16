/**
 * Model files: downloaded once from MODELS_URL (a Hugging Face repo, or models/ on the app's own server),
 * checked against the SHA-256 in its manifest.json, and kept in the browser's Origin Private File System
 * (Cache Storage refuses single entries this large). Stored names start with part of the checksum, so an
 * updated model is never mixed up with an old one.
 */
export interface ManifestEntry {
  file: string;
  bytes: number;
  sha256: string;
}
export type Manifest = Record<string, ManifestEntry>;

const DIR = "splitwave-models";
const CACHE = "splitwave-models-v1";   // fallback for browsers without OPFS
/**
 * Where the models live, set at build time in web/.env (VITE_MODELS_URL): e.g.
 * https://huggingface.co/<user>/<repo>/resolve/main/ (Hugging Face sends the cross-origin headers browsers
 * need, and the page's isolation headers allow CORS downloads). Empty = models/ on the app's own server.
 */
const MODELS_URL = (import.meta.env.VITE_MODELS_URL || "models/").replace(/\/?$/, "/");
const base = () => new URL(MODELS_URL, self.location.origin + import.meta.env.BASE_URL);
const get = (url: URL, init: RequestInit = {}) => fetch(url, { mode: "cors", credentials: "omit", ...init });

/** Host name shown in messages ("huggingface.co", or this site). */
export const modelHost = () => base().host;
const storedName = (m: ManifestEntry) => `${m.sha256.slice(0, 16)}-${m.file}`;

let manifest: Promise<Manifest> | null = null;
export function getManifest(): Promise<Manifest> {
  manifest ??= get(new URL("manifest.json", base()), { cache: "no-cache" }).then((r) => {
    if (!r.ok) throw new Error(`model list unavailable from ${base().host} (${r.status})`);
    return r.json();
  });
  return manifest;
}

async function sha256(buf: Uint8Array): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", buf as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function opfs(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(DIR, { create: true });
  } catch {
    return null;
  }
}

async function readStored(m: ManifestEntry): Promise<Uint8Array | null> {
  const dir = await opfs();
  if (dir) {
    try {
      const f = await (await dir.getFileHandle(storedName(m))).getFile();
      if (f.size === m.bytes) return new Uint8Array(await f.arrayBuffer());
    } catch { /* not stored */ }
  }
  try {
    const hit = await (await caches.open(CACHE)).match(new URL(storedName(m), base()));
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch { /* no Cache Storage */ }
  return null;
}

type SyncHandle = { truncate(n: number): void; write(b: Uint8Array, o: { at: number }): number; flush(): void; close(): void };

async function store(m: ManifestEntry, buf: Uint8Array): Promise<void> {
  const dir = await opfs();
  if (dir) {
    try {
      // older versions of the same model
      for await (const name of (dir as unknown as { keys(): AsyncIterable<string> }).keys()) {
        if (name.endsWith(`-${m.file}`) && name !== storedName(m)) await dir.removeEntry(name).catch(() => {});
      }
      const fh = await dir.getFileHandle(storedName(m), { create: true });
      if ("createWritable" in fh) {
        const w = await fh.createWritable();
        await w.write(buf as Uint8Array<ArrayBuffer>);
        await w.close();
      } else {                                    // Safari before createWritable: workers have sync handles
        const h = await (fh as unknown as { createSyncAccessHandle(): Promise<SyncHandle> }).createSyncAccessHandle();
        h.truncate(0);
        h.write(buf, { at: 0 });
        h.flush();
        h.close();
      }
      return;
    } catch { /* fall back to Cache Storage */ }
  }
  try {
    await (await caches.open(CACHE)).put(new URL(storedName(m), base()), new Response(buf as Uint8Array<ArrayBuffer>));
  } catch { /* storage full or private window: works, just downloads again next time */ }
}

/** Whether a model is already downloaded. */
export async function isCached(id: string): Promise<boolean> {
  const m = (await getManifest())[id];
  if (!m) return false;
  const dir = await opfs();
  if (dir) {
    try {
      if ((await (await dir.getFileHandle(storedName(m))).getFile()).size === m.bytes) return true;
    } catch { /* not there */ }
  }
  try {
    return !!(await (await caches.open(CACHE)).match(new URL(storedName(m), base())));
  } catch {
    return false;
  }
}

export async function loadModel(id: string, onProgress: (loaded: number, total: number) => void): Promise<Uint8Array> {
  const m = (await getManifest())[id];
  if (!m) throw new Error(`unknown model ${id}`);
  const stored = await readStored(m);
  if (stored) {
    onProgress(stored.length, stored.length);
    return stored;
  }
  const res = await get(new URL(m.file, base()));
  if (!res.ok || !res.body) throw new Error(`couldn't download ${m.file} from ${base().host} (${res.status})`);
  const total = m.bytes;
  const buf = new Uint8Array(total);
  const reader = res.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > total) throw new Error(`${m.file} is larger than expected`);
    buf.set(value, loaded);
    loaded += value.length;
    onProgress(loaded, total);
  }
  if (loaded !== total) throw new Error(`${m.file} downloaded incompletely (${loaded} of ${total} bytes)`);
  if ((await sha256(buf)) !== m.sha256) throw new Error(`${m.file} failed its checksum; try again`);
  await store(m, buf);
  return buf;
}
