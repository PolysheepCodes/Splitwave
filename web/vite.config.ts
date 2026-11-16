import { createReadStream, existsSync, statSync } from "node:fs";
import { resolve, sep } from "node:path";
import { defineConfig, type Plugin } from "vite";

// The page must be cross-origin isolated for multithreaded WebAssembly (same headers as server/serve.mjs).
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};

/** In development, serve ../models at /models like the production server does. */
function models(): Plugin {
  const root = resolve(import.meta.dirname, "..", "models");
  return {
    name: "splitwave-models",
    configureServer(server) {
      server.middlewares.use("/models", (req, res, next) => {
        const file = resolve(root, "." + decodeURIComponent((req.url || "/").split("?")[0]));
        if (!file.startsWith(root + sep) || !existsSync(file) || statSync(file).isDirectory()) return next();
        res.setHeader("Content-Type", file.endsWith(".json") ? "application/json" : "application/octet-stream");
        res.setHeader("Content-Length", statSync(file).size);
        for (const [k, v] of Object.entries(isolation)) res.setHeader(k, v);
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [models()],
  server: { headers: isolation, port: 5173 },
  preview: { headers: isolation },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022", chunkSizeWarningLimit: 2000 },
});
