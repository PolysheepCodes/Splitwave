#!/usr/bin/env node
// Splitwave's server: serves the built web app and the model files. Nothing is computed here; separation
// runs in the visitor's browser. No dependencies: `node server/serve.mjs` (PORT, APP_DIR, MODELS_DIR).
//
// Headers that matter:
//  - Cross-Origin-Opener-Policy / -Embedder-Policy: make the page "cross-origin isolated", which browsers
//    require before WebAssembly may use several CPU threads (SharedArrayBuffer).
//  - Range requests + long caching for models: downloads can resume and are fetched once per visitor.
import { createReadStream, statSync, existsSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const PORT = Number(process.env.PORT || 8080);
const APP_DIR = resolve(process.env.APP_DIR || join(here, "..", "web", "dist"));
const MODELS_DIR = resolve(process.env.MODELS_DIR || join(here, "..", "models"));

const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8", ".wasm": "application/wasm",
  ".onnx": "application/octet-stream", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
  ".webmanifest": "application/manifest+json", ".txt": "text/plain; charset=utf-8",
};

function inside(root, urlPath) {
  const p = normalize(join(root, decodeURIComponent(urlPath)));
  return p === root || p.startsWith(root + sep) ? p : null;
}

createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  let file;
  if (url.pathname.startsWith("/models/")) file = inside(MODELS_DIR, url.pathname.slice("/models".length));
  else file = inside(APP_DIR, url.pathname === "/" ? "/index.html" : url.pathname);
  if (file && existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
  if (!file || !existsSync(file)) {
    res.writeHead(404, { "Content-Type": "text/plain" }).end("not found");
    return;
  }
  const { size, mtime } = statSync(file);
  const headers = {
    "Content-Type": TYPES[extname(file)] || "application/octet-stream",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Accept-Ranges": "bytes",
    "Last-Modified": mtime.toUTCString(),
    // model files and hashed build assets never change under the same name; the page itself is re-checked
    "Cache-Control": url.pathname.startsWith("/models/") || url.pathname.startsWith("/assets/")
      ? "public, max-age=31536000, immutable" : "no-cache",
  };
  const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || "");
  if (range) {
    const start = range[1] ? Number(range[1]) : size - Number(range[2]);
    const end = range[1] && range[2] ? Number(range[2]) : size - 1;
    if (start >= size || end >= size || start > end) {
      res.writeHead(416, { ...headers, "Content-Range": `bytes */${size}` }).end();
      return;
    }
    res.writeHead(206, { ...headers, "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": end - start + 1 });
    if (req.method === "HEAD") return res.end();
    createReadStream(file, { start, end }).pipe(res);
    return;
  }
  res.writeHead(200, { ...headers, "Content-Length": size });
  if (req.method === "HEAD") return res.end();
  createReadStream(file).pipe(res);
}).listen(PORT, () => console.log(`Splitwave on http://localhost:${PORT}  (app ${APP_DIR}, models ${MODELS_DIR})`));
