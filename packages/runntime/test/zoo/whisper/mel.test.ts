import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  createLogMel,
  createPowerSpectrum,
  MEL_BINS,
  MEL_FRAMES,
} from '../../../src/zoo/whisper/mel.ts';
import { randomF32 } from '../../gpu.ts';

const REF = new URL('../../../.weights/whisper-small/ref/', import.meta.url);

/** |Σ x[n] e^(−2πikn/N)|², straight from the DFT formula. */
function naivePower(x: Float64Array): Float64Array {
  const n = x.length;
  return Float64Array.from({ length: n / 2 + 1 }, (_, k) => {
    let re = 0;
    let im = 0;
    for (let j = 0; j < n; j++) {
      re += x[j]! * Math.cos((2 * Math.PI * k * j) / n);
      im -= x[j]! * Math.sin((2 * Math.PI * k * j) / n);
    }
    return re * re + im * im;
  });
}

describe('whisper log-mel', () => {
  it.each([
    { name: 'random samples', frame: Float64Array.from(randomF32(400, 3)) },
    { name: 'an impulse', frame: Float64Array.from({ length: 400 }, (_, i) => (i === 0 ? 1 : 0)) },
    {
      name: 'a 440 Hz tone',
      frame: Float64Array.from({ length: 400 }, (_, i) =>
        Math.sin((2 * Math.PI * 440 * i) / 16000),
      ),
    },
  ])('power spectrum of $name matches the DFT', ({ frame }) => {
    const got = new Float64Array(201);
    createPowerSpectrum()(frame, got);
    const want = naivePower(frame);
    for (let k = 0; k < 201; k++) expect(got[k]).toBeCloseTo(want[k]!, 8);
  });

  it('gives silence the floor value everywhere', () => {
    const mel = createLogMel()(new Float32Array(16000));
    // log10(1e-10) = −10, the max too, so (−10 + 4) / 4.
    expect(new Set(mel)).toEqual(new Set([-1.5]));
  });

  const manifest = new URL('manifest.json', REF);
  const clips: { id: string; mel: boolean }[] = existsSync(manifest)
    ? JSON.parse(readFileSync(manifest, 'utf8'))
    : [];
  // Hugging Face's WhisperFeatureExtractor output, dumped next to the weights.
  it.skipIf(clips.length === 0).each(clips.filter((c) => c.mel))(
    'matches transformers on $id',
    ({ id }) => {
      const read = (name: string) => {
        const b = readFileSync(new URL(name, REF));
        return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
      };
      const want = read(`${id}.mel.f32`);
      const got = createLogMel()(read(`${id}.f32`));
      expect(got.length).toBe(MEL_FRAMES * MEL_BINS);
      let worst = 0;
      for (let i = 0; i < got.length; i++) worst = Math.max(worst, Math.abs(got[i]! - want[i]!));
      // transformers runs the STFT in f32; this runs it in f64.
      expect(worst).toBeLessThan(1e-4);
    },
  );
});
