/// <reference types="vite/client" />
interface ImportMetaEnv {
  /** Where the model files and manifest.json are (see web/.env). */
  readonly VITE_MODELS_URL?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
