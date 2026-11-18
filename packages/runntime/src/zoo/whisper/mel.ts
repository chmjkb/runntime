/** Whisper's log-mel spectrogram on the CPU: 16 kHz mono audio, zero-padded
 *  to 30 s, to 80 mel bins × 3000 frames, the same math as Hugging Face's
 *  WhisperFeatureExtractor (periodic Hann window, centered STFT with reflect
 *  padding, power spectrum, Slaney mel filters, log10, clamp to max − 8,
 *  then (x + 4) / 4).
 *
 *  Frames that hold only padding have zero power, so they are not
 *  transformed: a 5 s clip runs a sixth of the FFTs. */

/** Samples per window the model takes, 30 s at 16 kHz. */
export const WINDOW_SAMPLES = 480_000;
/** Mel frames per window: one every HOP samples, the last STFT frame dropped. */
export const MEL_FRAMES = 3000;
export const MEL_BINS = 80;
const SAMPLE_RATE = 16_000;
const N_FFT = 400;
const HOP = 160;
const FREQ_BINS = N_FFT / 2 + 1;
/** Power floor before log10, HF's mel_floor. */
const MEL_FLOOR = 1e-10;
/** The log spectrum is clamped to this many decades below its maximum. */
const DYNAMIC_RANGE = 8;

const HALF = N_FFT / 2;
/** Prime factors of HALF, the size of the complex FFT the real one packs into. */
const FACTORS = [2, 2, 2, 5, 5];

/** HF's Slaney mel scale: linear below 1 kHz, logarithmic above. */
function hzToMel(hz: number): number {
  const logStep = Math.log(6.4) / 27;
  return hz < 1000 ? (3 * hz) / 200 : 15 + Math.log(hz / 1000) / logStep;
}

function melToHz(mel: number): number {
  const logStep = Math.log(6.4) / 27;
  return mel < 15 ? (200 * mel) / 3 : 1000 * Math.exp(logStep * (mel - 15));
}

/** Slaney-normalized triangular filters, [MEL_BINS, FREQ_BINS] row-major. */
export function melFilters(): Float64Array {
  const melMax = hzToMel(SAMPLE_RATE / 2);
  const edges = Array.from({ length: MEL_BINS + 2 }, (_, i) =>
    melToHz((melMax * i) / (MEL_BINS + 1)),
  );
  const out = new Float64Array(MEL_BINS * FREQ_BINS);
  for (let m = 0; m < MEL_BINS; m++) {
    const [lo, mid, hi] = [edges[m]!, edges[m + 1]!, edges[m + 2]!];
    const norm = 2 / (hi - lo);
    for (let k = 0; k < FREQ_BINS; k++) {
      const f = (k * SAMPLE_RATE) / N_FFT;
      const weight = Math.max(0, Math.min((f - lo) / (mid - lo), (hi - f) / (hi - mid)));
      out[m * FREQ_BINS + k] = weight * norm;
    }
  }
  return out;
}

/** Complex DFT of `n` points read from `src` at `offset` with `stride`,
 *  written contiguously to `dst` at `at`. Mixed-radix, decimation in time;
 *  `twiddle` holds e^(−2πij/HALF). */
function fft(
  n: number,
  level: number,
  re: Float64Array,
  im: Float64Array,
  offset: number,
  stride: number,
  dstRe: Float64Array,
  dstIm: Float64Array,
  at: number,
  twiddle: { re: Float64Array; im: Float64Array },
  scratch: Float64Array,
): void {
  if (n === 1) {
    dstRe[at] = re[offset]!;
    dstIm[at] = im[offset]!;
    return;
  }
  const p = FACTORS[level]!;
  const m = n / p;
  for (let r = 0; r < p; r++) {
    fft(
      m,
      level + 1,
      re,
      im,
      offset + r * stride,
      stride * p,
      dstRe,
      dstIm,
      at + r * m,
      twiddle,
      scratch,
    );
  }
  // X[k + q·m] = Σ_r W_n^(r·(k + q·m)) · Y_r[k], with W_n = W_HALF^(HALF / n).
  const step = HALF / n;
  for (let k = 0; k < m; k++) {
    for (let q = 0; q < p; q++) {
      let sr = 0;
      let si = 0;
      for (let r = 0; r < p; r++) {
        const e = ((r * (k + q * m)) % n) * step;
        const wr = twiddle.re[e]!;
        const wi = twiddle.im[e]!;
        const yr = dstRe[at + r * m + k]!;
        const yi = dstIm[at + r * m + k]!;
        sr += wr * yr - wi * yi;
        si += wr * yi + wi * yr;
      }
      scratch[2 * q] = sr;
      scratch[2 * q + 1] = si;
    }
    for (let q = 0; q < p; q++) {
      dstRe[at + k + q * m] = scratch[2 * q]!;
      dstIm[at + k + q * m] = scratch[2 * q + 1]!;
    }
  }
}

/** Power spectrum |X[k]|², k = 0..N_FFT/2, of one real frame of N_FFT
 *  samples. Packs even and odd samples into one complex FFT of HALF points. */
export function createPowerSpectrum(): (frame: Float64Array, out: Float64Array) => void {
  const twiddle = { re: new Float64Array(HALF), im: new Float64Array(HALF) };
  for (let j = 0; j < HALF; j++) {
    twiddle.re[j] = Math.cos((-2 * Math.PI * j) / HALF);
    twiddle.im[j] = Math.sin((-2 * Math.PI * j) / HALF);
  }
  const zRe = new Float64Array(HALF);
  const zIm = new Float64Array(HALF);
  const fRe = new Float64Array(HALF);
  const fIm = new Float64Array(HALF);
  const scratch = new Float64Array(2 * Math.max(...FACTORS));
  return (frame, out) => {
    for (let n = 0; n < HALF; n++) {
      zRe[n] = frame[2 * n]!;
      zIm[n] = frame[2 * n + 1]!;
    }
    fft(HALF, 0, zRe, zIm, 0, 1, fRe, fIm, 0, twiddle, scratch);
    for (let k = 0; k <= HALF; k++) {
      const a = k % HALF;
      const b = (HALF - k) % HALF;
      // Even part E = (Z[k] + conj Z[−k]) / 2, odd part O = (Z[k] − conj Z[−k]) / 2i.
      const eRe = (fRe[a]! + fRe[b]!) / 2;
      const eIm = (fIm[a]! - fIm[b]!) / 2;
      const oRe = (fIm[a]! + fIm[b]!) / 2;
      const oIm = -(fRe[a]! - fRe[b]!) / 2;
      const angle = (-2 * Math.PI * k) / N_FFT;
      const wr = Math.cos(angle);
      const wi = Math.sin(angle);
      const xr = eRe + wr * oRe - wi * oIm;
      const xi = eIm + wr * oIm + wi * oRe;
      out[k] = xr * xr + xi * xi;
    }
  };
}

/** Builds the feature extractor. Its tables are made once; each call
 *  allocates only its output. */
export function createLogMel(): (audio: Float32Array) => Float32Array {
  const filters = melFilters();
  const window = Float64Array.from(
    { length: N_FFT },
    (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT),
  );
  const power = createPowerSpectrum();
  const frame = new Float64Array(N_FFT);
  const spectrum = new Float64Array(FREQ_BINS);
  const silentLog = Math.log10(MEL_FLOOR);

  return (audio) => {
    const length = Math.min(audio.length, WINDOW_SAMPLES);
    // The padded signal, reflected at both ends of the 30 s window.
    const sample = (i: number): number => {
      const j = i < 0 ? -i : i >= WINDOW_SAMPLES ? 2 * (WINDOW_SAMPLES - 1) - i : i;
      return j < length ? audio[j]! : 0;
    };
    // Frames past this one see only zeros.
    const voiced = Math.min(MEL_FRAMES, Math.ceil((length + HALF) / HOP));
    // [MEL_BINS, MEL_FRAMES], the layout the encoder's conv reads transposed.
    const logMel = new Float64Array(MEL_BINS * MEL_FRAMES).fill(silentLog);
    let max = silentLog;
    for (let f = 0; f < voiced; f++) {
      const start = f * HOP - HALF;
      for (let n = 0; n < N_FFT; n++) frame[n] = sample(start + n) * window[n]!;
      power(frame, spectrum);
      for (let m = 0; m < MEL_BINS; m++) {
        let sum = 0;
        const row = m * FREQ_BINS;
        for (let k = 0; k < FREQ_BINS; k++) sum += filters[row + k]! * spectrum[k]!;
        const v = Math.log10(Math.max(sum, MEL_FLOOR));
        logMel[m * MEL_FRAMES + f] = v;
        if (v > max) max = v;
      }
    }
    // Time-major [MEL_FRAMES, MEL_BINS] for the conv stem.
    const out = new Float32Array(MEL_FRAMES * MEL_BINS);
    const floor = max - DYNAMIC_RANGE;
    for (let m = 0; m < MEL_BINS; m++) {
      for (let f = 0; f < MEL_FRAMES; f++) {
        out[f * MEL_BINS + m] = (Math.max(logMel[m * MEL_FRAMES + f]!, floor) + 4) / 4;
      }
    }
    return out;
  };
}
