/** Whisper decoder: token embedding plus learned positions, then pre-norm
 *  layers of causal self-attention, cross-attention over the encoder states
 *  and a GELU MLP, then a final LN. The token embedding doubles as the tied
 *  LM head.
 *
 *  Decoding runs one token per step against the per-layer KV cache, so a
 *  step costs the same at every position. Module fields are named like the
 *  checkpoint. */

import { add, argmaxDot, gatherRowsFrom, nn, slice, type Value } from '../../core/index.ts';
import type { WhisperConfig } from './config.ts';
import { WhisperCrossAttention, WhisperSelfAttention, type KVCache } from './attention.ts';

export class WhisperDecoderLayer extends nn.Module<[Value, KVCache, KVCache, number], Value> {
  readonly self_attn_layer_norm: nn.LayerNorm;
  readonly self_attn: WhisperSelfAttention;
  readonly encoder_attn_layer_norm: nn.LayerNorm;
  readonly encoder_attn: WhisperCrossAttention;
  readonly final_layer_norm: nn.LayerNorm;
  readonly fc1: nn.Linear;
  readonly fc2: nn.Linear;
  constructor(cfg: WhisperConfig) {
    super();
    this.self_attn_layer_norm = new nn.LayerNorm(cfg.dModel, 1e-5, { bias: true });
    this.self_attn = new WhisperSelfAttention(cfg, true);
    this.encoder_attn_layer_norm = new nn.LayerNorm(cfg.dModel, 1e-5, { bias: true });
    this.encoder_attn = new WhisperCrossAttention(cfg);
    this.final_layer_norm = new nn.LayerNorm(cfg.dModel, 1e-5, { bias: true });
    this.fc1 = new nn.Linear(cfg.dModel, cfg.ffn, { activation: 'gelu' });
    this.fc2 = new nn.Linear(cfg.ffn, cfg.dModel);
  }
  forward(x1: Value, cache: KVCache, cross: KVCache, t: number): Value {
    const h1 = this.self_attn.forwardStep(this.self_attn_layer_norm.forward(x1), cache, t, x1);
    const h2 = this.encoder_attn.forward(this.encoder_attn_layer_norm.forward(h1), cross, h1);
    return this.fc2.forward(this.fc1.forward(this.final_layer_norm.forward(h2)), { addend: h2 });
  }
}

export class WhisperDecoder extends nn.Module<
  [Value, number, readonly KVCache[], readonly KVCache[]],
  Value
> {
  readonly embed_tokens: nn.Embedding;
  readonly embed_positions: nn.Embedding;
  readonly layers: nn.ModuleList<WhisperDecoderLayer>;
  readonly layer_norm: nn.LayerNorm;
  constructor(cfg: WhisperConfig) {
    super();
    this.embed_tokens = new nn.Embedding(cfg.vocab, cfg.dModel);
    this.embed_positions = new nn.Embedding(cfg.decPositions, cfg.dModel);
    this.layers = new nn.ModuleList(
      Array.from({ length: cfg.decLayers }, () => new WhisperDecoderLayer(cfg)),
    );
    this.layer_norm = new nn.LayerNorm(cfg.dModel, 1e-5, { bias: true });
  }
  /** Each layer's cross-attention k and v over the encoder states. */
  precomputeCross(encoded: Value): KVCache[] {
    return this.layers.items.map((layer) => layer.encoder_attn.precompute(encoded));
  }
  /** The embedded token `id`, a CPU id or a [1, 1] f32 id on the GPU, at
   *  position `t`. */
  embed(id: number | Value, t: number): Value {
    const token =
      typeof id === 'number'
        ? this.embed_tokens.forward([id])
        : gatherRowsFrom(this.embed_tokens.weight.value, id);
    return add(token, slice(this.embed_positions.weight.value, 0, t, t + 1));
  }
  /** Hidden state [1, d] of the token `x1` at position `t`. */
  forward(x1: Value, t: number, caches: readonly KVCache[], cross: readonly KVCache[]): Value {
    let h = x1;
    this.layers.items.forEach((layer, i) => {
      h = layer.forward(h, caches[i]!, cross[i]!, t);
    });
    return this.layer_norm.forward(h);
  }
  /** Greedy next token id of `h1`, [1, 1] f32. `bias`, f32 [vocab], is added
   *  to the logits first, -1e30 where a token is suppressed. */
  nextTokenId(h1: Value, bias: Value): Value {
    return argmaxDot(this.embed_tokens.weight.value, h1, bias);
  }
}
