/** Whisper tokenizer, decode-only: the decoder's prompt is fixed special
 *  ids, so id → text is the whole job. tokenizers.js runs the Hub's
 *  byte-level BPE tokenizer.json with its special ids skipped. */

import { Tokenizer } from '@huggingface/tokenizers';
import { checkpointMismatch } from '../errors.ts';
import { END_OF_TEXT, START_OF_TRANSCRIPT } from './config.ts';

export interface WhisperTokenizer {
  /** Text of the ids, special ids skipped. */
  decode(ids: number[]): string;
}

const isBpeTokenizerJson = (json: unknown): json is object =>
  typeof json === 'object' &&
  json !== null &&
  'model' in json &&
  typeof json.model === 'object' &&
  json.model !== null &&
  'type' in json.model &&
  json.model.type === 'BPE';

/** Builds the tokenizer from Whisper's tokenizer.json. Throws
 *  CHECKPOINT_MISMATCH for any other file. */
export function createWhisperTokenizer(json: unknown): WhisperTokenizer {
  const bad = (why: string) => checkpointMismatch('whisper tokenizer', why);
  if (!isBpeTokenizerJson(json)) throw bad('expected a BPE tokenizer.json');
  const tokenizer = new Tokenizer(json, {});
  if (
    tokenizer.token_to_id('<|endoftext|>') !== END_OF_TEXT ||
    tokenizer.token_to_id('<|startoftranscript|>') !== START_OF_TRANSCRIPT
  ) {
    throw bad('special token ids are not the multilingual Whisper ones');
  }
  return {
    decode: (ids) =>
      tokenizer.decode(ids, { skip_special_tokens: true, clean_up_tokenization_spaces: false }),
  };
}
