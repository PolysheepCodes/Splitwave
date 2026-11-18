/// <reference types="vite/client" />
interface ImportMetaEnv {
  /** Where the model files and manifest.json are (see web/.env). */
  readonly VITE_MODELS_URL?: string;
  /** Donation page linked from the banner at the top; empty = no banner (see web/.env). */
  readonly VITE_DONATE_URL?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
