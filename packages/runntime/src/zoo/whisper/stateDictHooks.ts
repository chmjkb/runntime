/** Fuses the HF Whisper checkpoint's attention projections as lazy derived
 *  tensors; nothing is fetched here. Every other tensor binds under its own
 *  name, since the modules are named like the checkpoint.
 *
 *  Self-attention q|k|v row-stack into one `qkv`, cross-attention k|v into
 *  one `kv`. q's weight and bias absorb the 1/√headDim score scale. k has no
 *  bias, so its share of a fused bias is zeros, counted as 0 download bytes.
 *  Weights stay in torch layout for the core Linear hook to transpose. */

import {
  concatRows,
  derive,
  scaleWeight,
  type LazyStateDict,
  type LazyTensor,
} from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import type { WhisperConfig } from './config.ts';

function take(sd: LazyStateDict, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t) throw checkpointMismatch('whisper state dict', `missing tensor '${name}'`);
  sd.tensors.delete(name);
  return t;
}

/** Zeros shaped like the 1D `like`, read from nothing. */
function zerosLike(like: LazyTensor): LazyTensor {
  if (like.kind === 'quant') throw checkpointMismatch('whisper state dict', 'quantized bias');
  return derive([...like.shape], [like], () => new Float32Array(like.shape[0]!), 0);
}

/** Fuses `attn`'s projections named in `parts` into `<attn>.<fused>`,
 *  scaling q by `qScale`. */
function fuse(
  sd: LazyStateDict,
  attn: string,
  parts: readonly ('q' | 'k' | 'v')[],
  fused: string,
  qScale: number,
): void {
  const weights: LazyTensor[] = [];
  const biases: LazyTensor[] = [];
  for (const p of parts) {
    const w = take(sd, `${attn}.${p}_proj.weight`);
    weights.push(p === 'q' ? scaleWeight(w, qScale) : w);
    if (p === 'k') continue;
    const b = take(sd, `${attn}.${p}_proj.bias`);
    biases.push(p === 'q' ? scaleWeight(b, qScale) : b);
  }
  // k's bias slot sits right before v's.
  const v = biases.length - 1;
  biases.splice(v, 0, zerosLike(biases[v]!));
  sd.tensors.set(`${attn}.${fused}.weight`, concatRows(weights));
  sd.tensors.set(`${attn}.${fused}.bias`, concatRows(biases));
}

export function transformWhisperStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: WhisperConfig,
): void {
  const at = (name: string) => (prefix ? `${prefix}.${name}` : name);
  const qScale = 1 / Math.sqrt(cfg.headDim);
  // The LM head is the token embedding; some exports also store the tie.
  sd.tensors.delete(at('proj_out.weight'));
  for (let i = 0; i < cfg.encLayers; i++) {
    fuse(sd, at(`model.encoder.layers.${i}.self_attn`), ['q', 'k', 'v'], 'qkv', qScale);
  }
  for (let i = 0; i < cfg.decLayers; i++) {
    const layer = at(`model.decoder.layers.${i}`);
    fuse(sd, `${layer}.self_attn`, ['q', 'k', 'v'], 'qkv', qScale);
    const cross = `${layer}.encoder_attn`;
    sd.tensors.set(
      `${cross}.q_proj.weight`,
      scaleWeight(take(sd, `${cross}.q_proj.weight`), qScale),
    );
    sd.tensors.set(`${cross}.q_proj.bias`, scaleWeight(take(sd, `${cross}.q_proj.bias`), qScale));
    fuse(sd, cross, ['k', 'v'], 'kv', qScale);
  }
}
