// WAV and ZIP writers produce files other tools open.
import { writeFileSync } from "node:fs";
import { wav16, zip } from "../src/ui/audio.ts";
const n = 44100, l = new Float32Array(n), r = new Float32Array(n);
for (let i = 0; i < n; i++) { l[i] = 0.5 * Math.sin(2 * Math.PI * 440 * i / 44100); r[i] = -l[i]; }
const w = wav16([l, r]);
writeFileSync(process.argv[2] + "/t.wav", new Uint8Array(await w.arrayBuffer()));
const z = await zip([{ name: "a (Vocals).wav", blob: w }, { name: "b (Instrumental).wav", blob: w }]);
writeFileSync(process.argv[2] + "/t.zip", new Uint8Array(await z.arrayBuffer()));
console.log("wrote", w.size, z.size);
