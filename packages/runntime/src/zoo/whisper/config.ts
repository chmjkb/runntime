/** Whisper architecture config and the special token ids the decoder prompt
 *  and the suppression lists use, as in openai/whisper-small's config.json
 *  and generation_config.json. */

import type { LazyStateDict } from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';

export interface WhisperConfig {
  readonly dModel: number;
  readonly ffn: number;
  readonly encLayers: number;
  readonly decLayers: number;
  readonly heads: number;
  readonly headDim: number;
  readonly melBins: number;
  /** Encoder frames per 30 s window. */
  readonly encPositions: number;
  /** Decoder positions, prompt included. */
  readonly decPositions: number;
  readonly vocab: number;
}

export const WHISPER_SMALL: WhisperConfig = Object.freeze({
  dModel: 768,
  ffn: 3072,
  encLayers: 12,
  decLayers: 12,
  heads: 12,
  headDim: 64,
  melBins: 80,
  encPositions: 1500,
  decPositions: 448,
  vocab: 51865,
});

export const START_OF_TRANSCRIPT = 50258;
export const END_OF_TEXT = 50257;
export const TRANSCRIBE = 50359;
export const NO_TIMESTAMPS = 50363;

/** Language tokens of the languages the task takes. */
export const LANGUAGE_TOKENS = {
  en: 50259,
  de: 50261,
  es: 50262,
  fr: 50265,
  pt: 50267,
  pl: 50269,
  it: 50274,
} as const;

export type WhisperLanguage = keyof typeof LANGUAGE_TOKENS;

/** Ids the model never emits, generation_config's suppress_tokens. */
export const SUPPRESS_TOKENS: readonly number[] = [
  1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503,
  522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536,
  3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562,
  13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161,
  26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359,
  50360, 50361, 50362,
];

/** Ids suppressed only as the first generated token, begin_suppress_tokens:
 *  a bare space and an empty transcript. */
export const BEGIN_SUPPRESS_TOKENS: readonly number[] = [220, END_OF_TEXT];

/** The preset whose width matches the checkpoint's token embedding. */
export function presetFromStateDict(sd: LazyStateDict): WhisperConfig {
  const embedding = sd.tensors.get('model.decoder.embed_tokens.weight');
  if (embedding === undefined) {
    throw checkpointMismatch('whisper weights', 'no token embedding, not a Whisper checkpoint');
  }
  const [vocab, width] = embedding.shape;
  if (width !== WHISPER_SMALL.dModel || vocab !== WHISPER_SMALL.vocab) {
    throw checkpointMismatch(
      'whisper weights',
      `embedding [${embedding.shape.join(', ')}] matches no known size`,
    );
  }
  return WHISPER_SMALL;
}
