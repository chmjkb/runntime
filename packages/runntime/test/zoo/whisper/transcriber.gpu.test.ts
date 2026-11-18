/** Whisper small against Hugging Face transformers' greedy f32 output on
 *  the same clips, one to three per language. The weights and references
 *  are local files under `.weights/whisper-small/`, written by
 *  `dump_whisper_ref.py`; without them the suite skips. */
import tgpu from 'typegpu';
import { describe, expect, it } from 'vitest';
import { initRunntime } from '../../../src/core/index.ts';
import { fetchJson, openWeights } from '../../../src/zoo/load.ts';
import { createSpeechToText } from '../../../src/zoo/tasks/audio/speechToText.ts';
import { createWhisperTokenizer } from '../../../src/zoo/whisper/tokenizer.ts';
import { createWhisperTranscriber } from '../../../src/zoo/whisper/transcriber.ts';
import { END_OF_TEXT, type WhisperLanguage } from '../../../src/zoo/whisper/config.ts';

const DIR = '/.weights/whisper-small';

interface Reference {
  id: string;
  language: WhisperLanguage;
  seconds: number;
  ids: number[];
  text: string;
}

const present = (await fetch(`${DIR}/ref/manifest.json`)).ok;
const refs: Reference[] = present ? await (await fetch(`${DIR}/ref/manifest.json`)).json() : [];
const audioOf = async (id: string) =>
  new Float32Array(await (await fetch(`${DIR}/ref/${id}.f32`)).arrayBuffer());

// The f32 token embedding is 159 MB, past the default 128 MiB binding limit.
const adapter = (await navigator.gpu.requestAdapter())!;
initRunntime(
  await tgpu.init({
    device: {
      optionalFeatures: ['subgroups', 'shader-f16'],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    },
  }),
);

describe.skipIf(!present)('whisper small', () => {
  // f32 matches every clip. f16 rounding may flip a near-tie token, and
  // int8 weights (`runntime quantize --bits 8 --f16`) a few more.
  it.for([
    { name: 'f32', file: 'model.safetensors', dtype: 'f32', misses: 0 },
    { name: 'f16', file: 'model.safetensors', dtype: 'f16', misses: 3 },
    { name: 'int8', file: 'model.int8.safetensors', dtype: 'f16', misses: 6 },
  ] as const)(
    'matches transformers on every clip, $name',
    { timeout: 120_000 },
    async ({ name, file, dtype, misses }) => {
      const tokenizer = createWhisperTokenizer(await fetchJson(`${DIR}/tokenizer.json`));
      const sd = await openWeights(`${DIR}/${file}`);
      const transcriber = await createWhisperTranscriber(sd, tokenizer, { weightDtype: dtype });
      try {
        let exact = 0;
        for (const ref of refs) {
          const got = await transcriber.transcribe(await audioOf(ref.id), ref.language);
          // transformers returns the ids without the EOS.
          const same = got.ids.filter((id) => id !== END_OF_TEXT).join() === ref.ids.join();
          if (same) exact++;
          console.log(
            `${name} ${ref.id} ${same ? 'exact' : 'DIFF'} mel ${got.timings.melMs.toFixed(0)} ms, ` +
              `encode ${got.timings.encodeMs.toFixed(0)} ms, decode ${got.timings.decodeMs.toFixed(0)} ms ` +
              `/ ${got.timings.steps} steps\n  ours: ${got.text}\n  hf:   ${ref.text}`,
          );
        }
        expect(exact).toBeGreaterThanOrEqual(refs.length - misses);
      } finally {
        transcriber.dispose();
      }
    },
  );
});

describe.skipIf(!present)('createSpeechToText with whisper', () => {
  it('transcribes each language through the task runner', { timeout: 120_000 }, async () => {
    const runner = await createSpeechToText({
      arch: 'whisper',
      modelPath: `${DIR}/model.safetensors`,
      tokenizerPath: `${DIR}/tokenizer.json`,
    });
    try {
      for (const language of new Set(refs.map((r) => r.language))) {
        const ref = refs.find((r) => r.language === language)!;
        expect(await runner.transcribe(await audioOf(ref.id), { language })).toBe(ref.text);
      }
      await expect(
        // @ts-expect-error: not a language the runner takes.
        runner.transcribe(new Float32Array(16000), { language: 'xx' }),
      ).rejects.toMatchObject({ code: 'INVALID_ARGUMENT' });
    } finally {
      runner.dispose();
    }
  });
});
