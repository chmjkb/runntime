/** Whisper transcriber: load once, then each transcribe(audio) turns every
 *  30 s window into log-mel frames on the CPU, encodes it once and runs a
 *  greedy KV-cached decode from the prompt
 *  <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>.
 *  Longer audio is split on quiet points, each window its own utterance.
 *
 *  Decoding runs in bursts: within a burst each step's argmax id feeds the
 *  next step's embedding on device, so the GPU runs the whole burst with no
 *  CPU round-trip. A burst may overshoot EOS; those ids are discarded. The
 *  prompt goes through the same steps, so there is no separate prefill.
 *
 *  Every shape is fixed by the 30 s window and the 448-position decoder, so
 *  nothing recompiles across calls, and the caches are allocated once.
 *  Calls share those buffers, so transcribe() serializes them. */

import {
  defaultRoot,
  evalValues,
  gpuExecutor,
  RunntimeError,
  tensor,
  writeRows,
  type GpuPerfSink,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import type { LoadOptions } from '../load.ts';
import { splitIntoChunks, type ChunkOpts } from '../moonshine/chunking.ts';
import type { KVCache } from './attention.ts';
import {
  BEGIN_SUPPRESS_TOKENS,
  END_OF_TEXT,
  LANGUAGE_TOKENS,
  NO_TIMESTAMPS,
  presetFromStateDict,
  START_OF_TRANSCRIPT,
  SUPPRESS_TOKENS,
  TRANSCRIBE,
  type WhisperLanguage,
} from './config.ts';
import { createLogMel, MEL_BINS, MEL_FRAMES } from './mel.ts';
import { WhisperModel } from './model.ts';
import type { WhisperTokenizer } from './tokenizer.ts';

export interface WhisperResult {
  /** Generated ids, prompt excluded, EOS included when reached. */
  ids: number[];
  text: string;
  timings: { melMs: number; encodeMs: number; decodeMs: number; steps: number };
}

export interface WhisperTranscriber {
  /** Calls run one at a time. The decode of each window stops at EOS, at
   *  the decoder's 448 positions, or after `tokensPerSecond` tokens per
   *  second of audio, default 15. */
  transcribe(
    audio: Float32Array,
    language: WhisperLanguage,
    tokensPerSecond?: number,
  ): Promise<WhisperResult>;
  dispose(): void;
}

export interface WhisperPerf {
  burst?: number;
  gpuPerf?: GpuPerfSink;
}

const SAMPLE_RATE = 16_000;
/** Decode budget. Whisper's byte-level BPE spends more tokens per word
 *  than Moonshine's, most in Polish. */
const TOKENS_PER_SECOND = 15;
/** Floor of the decode budget, so a one-word clip still gets its tokens. */
const MIN_TOKENS = 16;
const BURST = 8;
/** Longest window before a split. splitIntoChunks lets the last piece run
 *  1 s past it, so every piece fits the model's 30 s. */
const MAX_CHUNK_SECONDS = 29;
const SPLIT_SEARCH_SECONDS = 5;
/** Logit bias of a suppressed token. */
const SUPPRESSED = -1e30;

export async function createWhisperTranscriber(
  sd: LazyStateDict,
  tokenizer: WhisperTokenizer,
  opts: {
    weightDtype?: 'f32' | 'f16';
    onProgress?: LoadOptions['onProgress'];
    perf?: WhisperPerf;
  } = {},
): Promise<WhisperTranscriber> {
  const root = defaultRoot();
  if (sd.metadata['model'] && sd.metadata['model'] !== 'whisper') {
    throw checkpointMismatch(
      'whisper weights',
      `file is for model '${sd.metadata['model']}', expected 'whisper'`,
    );
  }
  const cfg = presetFromStateDict(sd);
  const burstMax = Math.max(1, opts.perf?.burst ?? BURST);
  const ex = gpuExecutor(root, { perf: opts.perf?.gpuPerf });

  const half = opts.weightDtype === 'f16';
  const model = new WhisperModel(cfg);
  if (half) model.half(root);
  try {
    await model.loadStateDict(sd, { root, onProgress: opts.onProgress });
  } catch (err) {
    model.dispose();
    throw err;
  }
  const { encoder, decoder } = model.model;
  const logMel = createLogMel();

  const actTensor = (data: Float32Array, dims: [number, number]): Value =>
    tensor(root, data, { elems: data.length, dtype: half ? 'f16' : 'f32', dims });

  const kvInit = new Float32Array(cfg.decPositions * cfg.dModel);
  const caches: KVCache[] = Array.from({ length: cfg.decLayers }, () => ({
    k: actTensor(kvInit, [cfg.decPositions, cfg.dModel]),
    v: actTensor(kvInit, [cfg.decPositions, cfg.dModel]),
  }));
  // Per-position greedy ids, read back once per burst.
  const idsBuf = tensor(root, new Float32Array(cfg.decPositions), [cfg.decPositions, 1]);
  const biasOf = (ids: readonly number[]): Value => {
    const bias = new Float32Array(cfg.vocab);
    for (const id of ids) bias[id] = SUPPRESSED;
    return tensor(root, bias, { elems: cfg.vocab, dtype: 'f32', dims: [cfg.vocab] });
  };
  const suppress = biasOf(SUPPRESS_TOKENS);
  const suppressFirst = biasOf([...SUPPRESS_TOKENS, ...BEGIN_SUPPRESS_TOKENS]);

  const transcribeWindow = async (
    audio: Float32Array,
    language: WhisperLanguage,
    tokensPerSecond: number,
  ): Promise<WhisperResult> => {
    const t0 = performance.now();
    const mel = actTensor(logMel(audio), [MEL_FRAMES, MEL_BINS]);
    const t1 = performance.now();
    const cross = decoder.precomputeCross(encoder.forward(mel));
    evalValues(
      cross.flatMap((c) => [c.k, c.v]),
      ex,
    );
    mel.buffer.destroy();
    const encodeMs = performance.now() - t1;

    const prompt = [START_OF_TRANSCRIPT, LANGUAGE_TOKENS[language], TRANSCRIBE, NO_TIMESTAMPS];
    const budget = Math.max(MIN_TOKENS, Math.ceil((audio.length / SAMPLE_RATE) * tokensPerSecond));
    const maxIds = Math.min(cfg.decPositions, prompt.length + budget);
    const ids = [...prompt];
    let fed = 0; // positions already in the KV cache
    const t2 = performance.now();
    decode: while (ids.length < maxIds) {
      const burst = Math.min(burstMax, maxIds - 1 - fed);
      // One DAG for the burst: a step past the known ids embeds the id the
      // step before it picked, on device.
      const idRows: Value[] = [];
      let picked: Value | undefined;
      for (let t = fed; t < fed + burst; t++) {
        const x = decoder.embed(t < ids.length ? ids[t]! : picked!, t);
        const h = decoder.forward(x, t, caches, cross);
        picked = decoder.nextTokenId(h, t === prompt.length - 1 ? suppressFirst : suppress);
        idRows.push(writeRows(idsBuf, picked, t));
      }
      evalValues(idRows, ex);
      const data = await ex.readback(idsBuf.buffer, {
        elems: fed + burst,
        dtype: 'f32',
        dims: [fed + burst, 1],
      });
      fed += burst;
      // Row t holds the id at position t + 1.
      for (let t = ids.length - 1; t < fed; t++) {
        const id = data[t]!;
        ids.push(id);
        if (id === END_OF_TEXT) break decode;
      }
    }
    for (const { k, v } of cross) {
      ex.releaseOutput(k.buffer, k.shape);
      k.markReleased();
      ex.releaseOutput(v.buffer, v.shape);
      v.markReleased();
    }
    const generated = ids.slice(prompt.length);
    return {
      ids: generated,
      text: tokenizer.decode(generated),
      timings: {
        melMs: t1 - t0,
        encodeMs,
        decodeMs: performance.now() - t2,
        steps: fed,
      },
    };
  };

  const chunkOpts: ChunkOpts = {
    maxSeconds: MAX_CHUNK_SECONDS,
    searchSeconds: SPLIT_SEARCH_SECONDS,
    sampleRate: SAMPLE_RATE,
  };

  const transcribeAll = async (
    audio: Float32Array,
    language: WhisperLanguage,
    tokensPerSecond: number,
  ): Promise<WhisperResult> => {
    const results: WhisperResult[] = [];
    for (const piece of splitIntoChunks(audio, chunkOpts)) {
      results.push(await transcribeWindow(piece, language, tokensPerSecond));
    }
    if (results.length === 1) return results[0]!;
    return {
      ids: results.flatMap((r) => r.ids),
      text: results
        .map((r) => r.text.trim())
        .filter((t) => t.length > 0)
        .join(' '),
      timings: results.reduce(
        (sum, r) => ({
          melMs: sum.melMs + r.timings.melMs,
          encodeMs: sum.encodeMs + r.timings.encodeMs,
          decodeMs: sum.decodeMs + r.timings.decodeMs,
          steps: sum.steps + r.timings.steps,
        }),
        { melMs: 0, encodeMs: 0, decodeMs: 0, steps: 0 },
      ),
    };
  };

  // The caches and the id buffer are shared by every call, so calls run
  // one at a time: each waits for the previous one to settle.
  let inFlight: Promise<unknown> = Promise.resolve();
  let disposed = false;
  return {
    transcribe(audio, language, tokensPerSecond = TOKENS_PER_SECOND) {
      if (disposed) {
        return Promise.reject(new RunntimeError('RESOURCE_DISPOSED', 'transcriber is disposed'));
      }
      const job = inFlight.then(() => transcribeAll(audio, language, tokensPerSecond));
      inFlight = job.catch(() => undefined);
      return job;
    },
    dispose() {
      disposed = true;
      model.dispose();
      for (const v of [idsBuf, suppress, suppressFirst, ...caches.flatMap((c) => [c.k, c.v])]) {
        v.buffer.destroy();
      }
      ex.dispose();
    },
  };
}
