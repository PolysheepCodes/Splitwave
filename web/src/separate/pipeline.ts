/**
 * Splitwave's stem chain: the song -> vocals + instrumental, then vocals -> lead + backing vocals.
 * Mirrors UtauCover's StemSeparator on top of python-audio-separator, levels included, so the stems match:
 *  - a song that peaks above 0.98 is turned down to 0.98 first (decoded mp3s often peak above full scale,
 *    and MDX models then leave singing in the instrumental);
 *  - each model step sees the mix normalised to <= 0.9 peak, and every stem is capped at 0.9 peak;
 *  - MDX's second stem is the mix minus the first times a per-model compensation factor.
 */
import { mdxDemix, type MdxConfig } from "./mdx.ts";
import { roformerDemix } from "./roformer.ts";
import type { Net, Progress, Stereo } from "./types.ts";

export type Preset = "fast" | "best";
export type StemName = "vocals" | "instrumental" | "lead" | "backing";

export interface ModelInfo {
  id: string;
  file: string;
  kind: "mdx" | "roformer";
  /** which stem the network outputs; the other is the input minus it */
  output: "Vocals" | "Instrumental";
  mdx?: MdxConfig;
}

export const MODELS: Record<string, ModelInfo> = {
  mdx_voc_ft: { id: "mdx_voc_ft", file: "mdx_voc_ft.onnx", kind: "mdx", output: "Vocals",
                mdx: { nFft: 7680, dimF: 3072, compensate: 1.021 } },
  mdx_kara_2: { id: "mdx_kara_2", file: "mdx_kara_2.onnx", kind: "mdx", output: "Instrumental",
                mdx: { nFft: 5120, dimF: 2048, compensate: 1.065 } },
  bs_roformer: { id: "bs_roformer", file: "bs_roformer_fp16.onnx", kind: "roformer", output: "Vocals" },
  mel_karaoke: { id: "mel_karaoke", file: "mel_roformer_karaoke_fp16.onnx", kind: "roformer", output: "Vocals" },
};

export const PRESETS: Record<Preset, { vocals: string; karaoke: string }> = {
  fast: { vocals: "mdx_voc_ft", karaoke: "mdx_kara_2" },
  best: { vocals: "bs_roformer", karaoke: "mel_karaoke" },
};

export function peak(s: Stereo): number {
  let p = 0;
  for (const ch of s) for (let i = 0; i < ch.length; i++) { const v = Math.abs(ch[i]); if (v > p) p = v; }
  return p;
}

function scaled(s: Stereo, k: number): Stereo {
  return s.map((ch) => { const o = new Float32Array(ch.length); for (let i = 0; i < ch.length; i++) o[i] = ch[i] * k; return o; }) as Stereo;
}

/** spec_utils.normalize: turn down to max_peak if louder (never up). Returns a copy. */
function capPeak(s: Stereo, maxPeak = 0.9): Stereo {
  const p = peak(s);
  return p > maxPeak ? scaled(s, maxPeak / p) : s.map((ch) => ch.slice()) as Stereo;
}

function minus(a: Stereo, b: Stereo, k = 1): Stereo {
  return a.map((ch, c) => { const o = new Float32Array(ch.length); const bc = b[c]; for (let i = 0; i < ch.length; i++) o[i] = ch[i] - bc[i] * k; return o; }) as Stereo;
}

/** One model on one input: [what the network outputs, the rest]. */
export async function runModel(model: ModelInfo, input: Stereo, net: Net, progress?: Progress): Promise<[Stereo, Stereo]> {
  let first: Stereo, second: Stereo;
  if (model.kind === "mdx") {
    const cfg = model.mdx!;
    const p = peak(input);
    const mix = capPeak(input);
    // audio-separator multiplies by the peak measured *before* normalising (not by the amount it turned
    // the mix down); kept as is so the stems' levels match UtauCover's
    first = scaled(await mdxDemix(mix, net, cfg, progress), p);
    second = minus(mix, first, cfg.compensate);
  } else {
    const mix = capPeak(input);
    first = await roformerDemix(mix, net, {}, progress);
    second = minus(mix, first);
  }
  return [capPeak(first), capPeak(second)];
}

export interface Stage {
  model: ModelInfo;
  label: string;
}

/** The models a preset runs, in order. */
export function stages(preset: Preset, karaoke: boolean): Stage[] {
  const p = PRESETS[preset];
  const out: Stage[] = [{ model: MODELS[p.vocals], label: "Separating vocals" }];
  if (karaoke) out.push({ model: MODELS[p.karaoke], label: "Splitting lead and backing vocals" });
  return out;
}

/**
 * Run the chain. `netFor` loads a model's network (and is told when it's no longer needed), so only one
 * model is in memory at a time.
 */
export async function separate(song: Stereo, preset: Preset, karaoke: boolean,
                               netFor: (m: ModelInfo) => Promise<Net & { release?: () => Promise<void> | void }>,
                               progress: (stage: number, done: number, total: number) => void): Promise<Partial<Record<StemName, Stereo>>> {
  const p = peak(song);
  const input = p > 0.98 ? scaled(song, 0.98 / p) : song;
  const plan = stages(preset, karaoke);
  const stems: Partial<Record<StemName, Stereo>> = {};
  for (let i = 0; i < plan.length; i++) {
    const { model } = plan[i];
    const net = await netFor(model);
    try {
      const src = i === 0 ? input : stems.vocals!;
      const [out, rest] = await runModel(model, src, net, (d, t) => progress(i, d, t));
      if (i === 0) {
        stems.vocals = model.output === "Vocals" ? out : rest;
        stems.instrumental = model.output === "Vocals" ? rest : out;
      } else {
        // karaoke models: "Vocals" = the lead singer, "Instrumental" = everything else (backing vocals)
        stems.lead = model.output === "Vocals" ? out : rest;
        stems.backing = model.output === "Vocals" ? rest : out;
      }
    } finally {
      await net.release?.();
    }
  }
  return stems;
}
