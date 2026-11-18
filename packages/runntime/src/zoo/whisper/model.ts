/** The full Whisper module tree, named like the HF checkpoint:
 *  `model.encoder` and `model.decoder` under WhisperForConditionalGeneration.
 *  One loadStateDict binds the whole checkpoint. */

import { nn, type LazyStateDict } from '../../core/index.ts';
import type { WhisperConfig } from './config.ts';
import { WhisperDecoder } from './decoder.ts';
import { WhisperEncoder } from './encoder.ts';
import { transformWhisperStateDict } from './stateDictHooks.ts';

/** HF's WhisperModel: the encoder and decoder pair. */
export class WhisperEncoderDecoder extends nn.Module {
  readonly encoder: WhisperEncoder;
  readonly decoder: WhisperDecoder;
  constructor(cfg: WhisperConfig) {
    super();
    this.encoder = new WhisperEncoder(cfg);
    this.decoder = new WhisperDecoder(cfg);
  }
}

export class WhisperModel extends nn.Module {
  readonly model: WhisperEncoderDecoder;
  constructor(readonly cfg: WhisperConfig) {
    super();
    this.model = new WhisperEncoderDecoder(cfg);
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformWhisperStateDict(sd, prefix, this.cfg);
  }
}
