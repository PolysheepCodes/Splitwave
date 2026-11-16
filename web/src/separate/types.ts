/** Stereo audio at 44.1 kHz: [left, right]. */
export type Stereo = [Float32Array, Float32Array];

/** A separation network: takes one input tensor, returns its output tensor (both float32). */
export interface Net {
  run(input: Float32Array, dims: number[]): Promise<Float32Array>;
  /** how many run() calls it can usefully have in flight at once (CPU worker pools); default 1 */
  parallel?: number;
}

export type Progress = (done: number, total: number) => void;

export const SAMPLE_RATE = 44100;
