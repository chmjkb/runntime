/** Whisper encoder: a two-conv stem over the log-mel frames, learned
 *  sinusoidal positions, then pre-norm transformer layers and a final LN.
 *
 *      conv1 k=3 p=1 → gelu → conv2 k=3 s=2 p=1 → gelu → + positions
 *      LN → bidirectional MHA → residual, LN → GELU MLP → residual
 *
 *  Input is always one 30 s window, [3000, 80] mel frames to [1500, d], so
 *  every shape is fixed. The 1/√headDim score scale is folded into q at
 *  load. Module fields are named like the checkpoint. */

import { add, nn, type Value } from '../../core/index.ts';
import type { WhisperConfig } from './config.ts';
import { WhisperSelfAttention } from './attention.ts';

export class WhisperEncoderLayer extends nn.Module<[Value], Value> {
  readonly self_attn_layer_norm: nn.LayerNorm;
  readonly self_attn: WhisperSelfAttention;
  readonly final_layer_norm: nn.LayerNorm;
  readonly fc1: nn.Linear;
  readonly fc2: nn.Linear;
  constructor(cfg: WhisperConfig) {
    super();
    this.self_attn_layer_norm = new nn.LayerNorm(cfg.dModel, 1e-5, { bias: true });
    this.self_attn = new WhisperSelfAttention(cfg, false);
    this.final_layer_norm = new nn.LayerNorm(cfg.dModel, 1e-5, { bias: true });
    this.fc1 = new nn.Linear(cfg.dModel, cfg.ffn, { activation: 'gelu' });
    this.fc2 = new nn.Linear(cfg.ffn, cfg.dModel);
  }
  forward(x: Value): Value {
    const h = this.self_attn.forward(this.self_attn_layer_norm.forward(x), x);
    return this.fc2.forward(this.fc1.forward(this.final_layer_norm.forward(h)), { addend: h });
  }
}

export class WhisperEncoder extends nn.Module<[Value], Value> {
  readonly conv1: nn.Conv1d;
  readonly conv2: nn.Conv1d;
  readonly embed_positions: nn.Embedding;
  readonly layers: nn.ModuleList<WhisperEncoderLayer>;
  readonly layer_norm: nn.LayerNorm;
  constructor(cfg: WhisperConfig) {
    super();
    const d = cfg.dModel;
    this.conv1 = new nn.Conv1d(cfg.melBins, d, 3, { padding: 1, activation: 'gelu' });
    this.conv2 = new nn.Conv1d(d, d, 3, { stride: 2, padding: 1, activation: 'gelu' });
    this.embed_positions = new nn.Embedding(cfg.encPositions, d);
    this.layers = new nn.ModuleList(
      Array.from({ length: cfg.encLayers }, () => new WhisperEncoderLayer(cfg)),
    );
    this.layer_norm = new nn.LayerNorm(d, 1e-5, { bias: true });
  }
  /** Mel frames [3000, melBins] to encoder states [1500, d]. */
  forward(mel: Value): Value {
    const stem = this.conv2.forward(this.conv1.forward(mel));
    let h = add(stem, this.embed_positions.weight.value);
    for (const layer of this.layers.items) h = layer.forward(h);
    return this.layer_norm.forward(h);
  }
}
