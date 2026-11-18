/** Whisper attention blocks. q carries the 1/√headDim score scale, folded
 *  into its weight and bias at load, so the kernels take none. k has no bias
 *  in the checkpoint; its share of a fused bias is zero. */

import { chunk, nn, slice, writeRows, type Value } from '../../core/index.ts';
import type { WhisperConfig } from './config.ts';

/** One decoder layer's self-attention cache, [decPositions, d] each. */
export interface KVCache {
  k: Value;
  v: Value;
}

/** Self-attention, bidirectional in the encoder and causal in the decoder. */
export class WhisperSelfAttention extends nn.Module<[Value, Value?], Value> {
  /** q | k | v rows, fused at load, q scaled. */
  readonly qkv: nn.Linear;
  readonly mha: nn.MultiHeadAttention;
  readonly out_proj: nn.Linear;
  private readonly dModel: number;
  constructor(cfg: WhisperConfig, causal: boolean) {
    super();
    const { dModel, heads, headDim } = cfg;
    this.dModel = dModel;
    this.qkv = new nn.Linear(dModel, 3 * dModel);
    this.mha = new nn.MultiHeadAttention({
      qHeads: heads,
      kvHeads: heads,
      headDim,
      windowLeft: Infinity,
      windowRight: causal ? 0 : Infinity,
    });
    this.out_proj = new nn.Linear(dModel, dModel);
  }
  forward(x: Value, residual?: Value): Value {
    return this.out_proj.forward(this.mha.forwardPacked(this.qkv.forward(x)), { addend: residual });
  }
  /** One token at position `t`: its k and v rows go into the cache, and q
   *  attends over cache rows 0..t. */
  forwardStep(x1: Value, cache: KVCache, t: number, residual: Value): Value {
    const d = this.dModel;
    const qkv = this.qkv.forward(x1); // [1, 3·d], columns q|k|v
    const k = writeRows(cache.k, slice(qkv, 1, d, 2 * d), t); // [t+1, d] view of the cache
    const v = writeRows(cache.v, slice(qkv, 1, 2 * d, 3 * d), t);
    const ctx = this.mha.forward(slice(qkv, 1, 0, d), k, v, { qPosOffset: t });
    return this.out_proj.forward(ctx, { addend: residual });
  }
}

/** Decoder cross-attention over the encoder states. */
export class WhisperCrossAttention extends nn.Module<[Value, KVCache, Value], Value> {
  readonly q_proj: nn.Linear;
  /** k | v rows, fused at load. */
  readonly kv: nn.Linear;
  readonly mha: nn.MultiHeadAttention;
  readonly out_proj: nn.Linear;
  constructor(cfg: WhisperConfig) {
    super();
    const { dModel, heads, headDim } = cfg;
    this.q_proj = new nn.Linear(dModel, dModel);
    this.kv = new nn.Linear(dModel, 2 * dModel);
    this.mha = new nn.MultiHeadAttention({
      qHeads: heads,
      kvHeads: heads,
      headDim,
      windowLeft: Infinity,
      windowRight: Infinity,
    });
    this.out_proj = new nn.Linear(dModel, dModel);
  }
  /** The encoder states' k and v, computed once per window. */
  precompute(encoded: Value): KVCache {
    const [k, v] = chunk(this.kv.forward(encoded), 2, 1);
    return { k: k!, v: v! };
  }
  forward(x: Value, cross: KVCache, residual: Value): Value {
    const ctx = this.mha.forward(this.q_proj.forward(x), cross.k, cross.v);
    return this.out_proj.forward(ctx, { addend: residual });
  }
}
