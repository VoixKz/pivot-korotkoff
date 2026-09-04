/* Признаки кадров для браузера — перенос src/audio.py.
 *
 * Модель обучалась на признаках, посчитанных в scipy. Если здесь считать
 * иначе, на вход придёт другое распределение и предсказания будут мусором,
 * причём молча. Поэтому каждый шаг повторяет питоновский буквально, включая
 * тип паддинга и обработку краёв, а совпадение проверяется тестом parity.html.
 */

export const C = {
  SR: 8000, HOP: 160, WIN: 400, N_FFT: 512,
  N_BANDS: 40, MEL_LO: 20, MEL_HI: 500,
  PER_WIN: 200, IBI_LO: 16, IBI_HI: 80,
  PADLEN: 27,
  // butter(4, [20, 200], btype='band', fs=8000, output='sos') — значения взяты
  // из scipy как есть, через tools/dump_dsp_const.py. Вписывать их по памяти
  // нельзя: неверные коэффициенты дают правдоподобный, но другой фильтр,
  // и модель молча получает не те признаки.
  SOS: [
    [2.0889537334624572e-05, 4.1779074669249144e-05, 2.0889537334624572e-05, 1.0, -1.7802330336356316, 0.7955186000327864],
    [1.0, 2.0, 1.0, 1.0, -1.8843767789008412, 0.9068981721347162],
    [1.0, -2.0, 1.0, 1.0, -1.9671131251397846, 0.9674648069340666],
    [1.0, -2.0, 1.0, 1.0, -1.989575366632527, 0.9898313279158593],
  ],
  // scipy.signal.sosfilt_zi: начальные условия установившегося режима.
  // Каждая секция домножена на накопленное усиление предыдущих на нулевой
  // частоте — своей формулой это легко упустить, поэтому берём из scipy.
  ZI: [
    [0.005445584335301936, -0.004327792104941017],
    [0.9654279669587221, -0.8750359198530803],
    [-0.9708944408313588, 0.9708944408313588],
    [-0.0, 0.0],
  ],
  SAVGOL5: [-0.0857142857142857, 0.3428571428571428, 0.4857142857142857,
            0.3428571428571428, -0.0857142857142857],
};

/* ------------------------------------------------------------------ БПФ */
/* Итеративный radix-2. Массивы re/im меняются на месте. */
export function fft(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

/* ------------------------------------------------------- фильтр Баттерворта */
/* Прямая форма II транспонированная, как в scipy.signal.sosfilt. */
function sosfilt(x, sos, ziScaled) {
  const y = Float64Array.from(x);
  for (let s = 0; s < sos.length; s++) {
    const [b0, b1, b2, , a1, a2] = sos[s];
    let z1 = ziScaled ? ziScaled[s][0] : 0, z2 = ziScaled ? ziScaled[s][1] : 0;
    for (let i = 0; i < y.length; i++) {
      const xi = y[i];
      const yi = b0 * xi + z1;
      z1 = b1 * xi - a1 * yi + z2;
      z2 = b2 * xi - a2 * yi;
      y[i] = yi;
    }
  }
  return y;
}

function reflectPad(x, left, right) {
  const n = x.length, out = new Float64Array(left + n + right);
  for (let i = 0; i < left; i++) out[i] = x[left - i];              // без повтора края
  out.set(x, left);
  for (let i = 0; i < right; i++) out[left + n + i] = x[n - 2 - i];
  return out;
}

function oddPad(x, len) {
  /* scipy padtype='odd': отражение относительно крайнего значения. */
  const n = x.length, out = new Float64Array(len + n + len);
  for (let i = 0; i < len; i++) out[i] = 2 * x[0] - x[len - i];
  out.set(x, len);
  for (let i = 0; i < len; i++) out[len + n + i] = 2 * x[n - 1] - x[n - 2 - i];
  return out;
}

export function sosfiltfilt(x, sos = C.SOS, padlen = C.PADLEN) {
  if (x.length <= padlen) return Float64Array.from(x);
  const zi = C.ZI;
  const ext = oddPad(x, padlen);

  const scale = s => zi.map(z => [z[0] * s, z[1] * s]);
  let y = sosfilt(ext, sos, scale(ext[0]));
  y.reverse();
  y = sosfilt(y, sos, scale(y[0]));
  y.reverse();
  return y.slice(padlen, padlen + x.length);
}

/* --------------------------------------------------------------- огибающая */
/* scipy.signal.savgol_filter(win=5, poly=2, mode='interp'): в середине —
 * свёртка, а на двух крайних отсчётах с каждой стороны подставляется значение
 * параболы, подогнанной по первым (последним) пяти точкам. */
function fitEdge(v, atStart) {
  const n = 5, xs = [0, 1, 2, 3, 4];
  const ys = atStart ? Array.from(v.slice(0, n))
                     : Array.from(v.slice(v.length - n));
  let S0 = n, S1 = 0, S2 = 0, S3 = 0, S4 = 0, T0 = 0, T1 = 0, T2 = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i], x2 = x * x;
    S1 += x; S2 += x2; S3 += x2 * x; S4 += x2 * x2;
    T0 += y; T1 += x * y; T2 += x2 * y;
  }
  // Нормальные уравнения 3x3, метод Крамера.
  const d = S0 * (S2 * S4 - S3 * S3) - S1 * (S1 * S4 - S3 * S2) + S2 * (S1 * S3 - S2 * S2);
  const c0 = (T0 * (S2 * S4 - S3 * S3) - S1 * (T1 * S4 - S3 * T2) + S2 * (T1 * S3 - S2 * T2)) / d;
  const c1 = (S0 * (T1 * S4 - S3 * T2) - T0 * (S1 * S4 - S3 * S2) + S2 * (S1 * T2 - T1 * S2)) / d;
  const c2 = (S0 * (S2 * T2 - T1 * S3) - S1 * (S1 * T2 - T1 * S2) + T0 * (S1 * S3 - S2 * S2)) / d;
  return x => c0 + c1 * x + c2 * x * x;
}

export function savgol5(v) {
  const n = v.length;
  if (n < 5) return Float64Array.from(v);
  const out = new Float64Array(n), k = C.SAVGOL5;
  for (let i = 2; i < n - 2; i++) {
    out[i] = k[0] * v[i - 2] + k[1] * v[i - 1] + k[2] * v[i] + k[3] * v[i + 1] + k[4] * v[i + 2];
  }
  const pl = fitEdge(v, true), pr = fitEdge(v, false);
  out[0] = pl(0); out[1] = pl(1);
  out[n - 2] = pr(3); out[n - 1] = pr(4);
  return out;
}

export function rmsEnvelope(y, hop = C.HOP) {
  const n = Math.floor(y.length / hop);
  if (n === 0) return new Float64Array(0);
  const env = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = i * hop; j < (i + 1) * hop; j++) s += y[j] * y[j];
    env[i] = Math.sqrt(s / hop) + 1e-9;
  }
  const sm = n > 5 ? savgol5(env) : env;
  for (let i = 0; i < sm.length; i++) sm[i] = Math.max(sm[i], 1e-9);
  return sm;
}

/* ------------------------------------------------------------ периодичность */
export function periodicity(env) {
  const n = env.length, W = C.PER_WIN, lo = C.IBI_LO;
  if (n < lo + 4) return new Float64Array(n);

  const pad = reflectPad(env, W >> 1, W - (W >> 1));
  let nfft = 1; while (nfft < 2 * W - 1) nfft <<= 1;
  const out = new Float64Array(n);

  const med = median(env);
  let devs = new Float64Array(n);
  for (let i = 0; i < n; i++) devs[i] = Math.abs(env[i] - med);
  const scale = median(devs) + 1e-12;

  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  const hi = Math.min(C.IBI_HI, W);
  if (hi <= lo) return out;

  for (let t = 0; t < n; t++) {
    let mean = 0;
    for (let k = 0; k < W; k++) mean += pad[t + k];
    mean /= W;
    let sd = 0;
    re.fill(0); im.fill(0);
    for (let k = 0; k < W; k++) {
      const z = pad[t + k] - mean;
      re[k] = z; sd += z * z;
    }
    if (Math.sqrt(sd / W) < 0.05 * scale) { out[t] = 0; continue; }

    fft(re, im, false);
    for (let k = 0; k < nfft; k++) {
      const p = re[k] * re[k] + im[k] * im[k];
      re[k] = p; im[k] = 0;
    }
    fft(re, im, true);

    const denom = re[0] > 0 ? re[0] : 1.0;
    let best = 0;
    for (let lag = lo; lag < hi; lag++) best = Math.max(best, re[lag] / denom);
    out[t] = Math.max(best, 0);
  }
  return out;
}

function median(v) {
  const a = Array.from(v).sort((x, y) => x - y);
  if (!a.length) return 0;
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

/* ------------------------------------------------------- логарифмические полосы */
let _fb = null;
function filterbank() {
  if (_fb) return _fb;
  const nBins = C.N_FFT / 2 + 1, nb = C.N_BANDS;
  const edges = new Float64Array(nb + 2);
  const r = Math.log(C.MEL_HI / C.MEL_LO) / (nb + 1);
  for (let i = 0; i < nb + 2; i++) edges[i] = C.MEL_LO * Math.exp(r * i);

  const freqs = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) freqs[i] = (i * C.SR) / C.N_FFT;

  _fb = [];
  for (let b = 0; b < nb; b++) {
    const lo = edges[b], mid = edges[b + 1], hi = edges[b + 2];
    const row = new Float64Array(nBins);
    let sum = 0;
    for (let i = 0; i < nBins; i++) {
      const f = freqs[i];
      let v = 0;
      if (f >= lo && f <= mid) v = (f - lo) / Math.max(mid - lo, 1e-9);
      else if (f > mid && f <= hi) v = (hi - f) / Math.max(hi - mid, 1e-9);
      row[i] = v; sum += v;
    }
    if (sum > 0) for (let i = 0; i < nBins; i++) row[i] /= sum;
    _fb.push(row);
  }
  return _fb;
}

export function logBandEnergies(x) {
  const hop = C.HOP, win = C.WIN, nfft = C.N_FFT;
  const n = Math.floor(x.length / hop);
  if (n === 0) return [];
  const padded = reflectPad(x, win >> 1, win);
  const window = new Float64Array(win);
  for (let i = 0; i < win; i++) window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (win - 1));

  const fb = filterbank(), nBins = nfft / 2 + 1;
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  const out = [];
  for (let t = 0; t < n; t++) {
    re.fill(0); im.fill(0);
    const off = t * hop;
    for (let i = 0; i < win; i++) re[i] = padded[off + i] * window[i];
    fft(re, im, false);
    const power = new Float64Array(nBins);
    for (let i = 0; i < nBins; i++) power[i] = re[i] * re[i] + im[i] * im[i];

    const row = new Float64Array(C.N_BANDS);
    for (let b = 0; b < C.N_BANDS; b++) {
      let s = 0;
      const f = fb[b];
      for (let i = 0; i < nBins; i++) s += power[i] * f[i];
      row[b] = Math.log(s + 1e-10);
    }
    out.push(row);
  }
  return out;
}

/* ------------------------------------------------------------------ признаки */
export function features(x) {
  // Усиление снимается до полос: иначе пол 1e-10 под логарифмом оказывается
  // на разной относительной высоте у тихой и громкой записи.
  let rms = 0;
  for (let i = 0; i < x.length; i++) rms += x[i] * x[i];
  rms = Math.sqrt(rms / x.length) + 1e-12;
  const xn = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) xn[i] = x[i] / rms;

  const y = sosfiltfilt(xn);
  const env = rmsEnvelope(y);
  const per = periodicity(env);
  const bands = logBandEnergies(xn);

  const T = Math.min(env.length, per.length, bands.length);
  if (T === 0) return { data: new Float32Array(0), T: 0, env: [], per: [] };

  let bMean = 0, bCount = 0;
  for (let t = 0; t < T; t++) for (let b = 0; b < C.N_BANDS; b++) { bMean += bands[t][b]; bCount++; }
  bMean /= bCount;
  let bVar = 0;
  for (let t = 0; t < T; t++) for (let b = 0; b < C.N_BANDS; b++) {
    const d = bands[t][b] - bMean; bVar += d * d;
  }
  const bStd = Math.sqrt(bVar / bCount) + 1e-6;

  const logEnv = new Float64Array(T);
  for (let t = 0; t < T; t++) logEnv[t] = Math.log(env[t] + 1e-9);
  let eMean = 0;
  for (let t = 0; t < T; t++) eMean += logEnv[t];
  eMean /= T;
  let eVar = 0;
  for (let t = 0; t < T; t++) { const d = logEnv[t] - eMean; eVar += d * d; }
  const eStd = Math.sqrt(eVar / T) + 1e-6;

  const F = C.N_BANDS + 2;
  const data = new Float32Array(T * F);
  for (let t = 0; t < T; t++) {
    for (let b = 0; b < C.N_BANDS; b++) data[t * F + b] = (bands[t][b] - bMean) / bStd;
    data[t * F + C.N_BANDS] = (logEnv[t] - eMean) / eStd;
    data[t * F + C.N_BANDS + 1] = per[t];
  }
  return { data, T, env: Array.from(env.slice(0, T)), per: Array.from(per.slice(0, T)) };
}
