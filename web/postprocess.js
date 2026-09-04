/* Кривая покадровых вероятностей -> две границы. Перенос src/postprocess.py. */

const HOP_S = 0.02;

export function sigmoid(v) {
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = 1 / (1 + Math.exp(-v[i]));
  return out;
}

export function medfilt(p, winS = 0.5) {
  let k = Math.round(winS / HOP_S) | 1;
  if (p.length < k) return Float64Array.from(p);
  const half = k >> 1, n = p.length, out = new Float64Array(n);
  const buf = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      const idx = i - half + j;
      buf[j] = idx < 0 || idx >= n ? 0 : p[idx];   // scipy.medfilt дополняет нулями
    }
    const s = Array.from(buf).sort((a, b) => a - b);
    out[i] = s[half];
  }
  return out;
}

/* Аускультативный провал — не конец интервала: у части гипертоников тоны
 * пропадают в середине и возвращаются. Интервал задаётся внешними границами. */
export function closeGaps(mask, maxGapS = 3.0) {
  const m = Array.from(mask);
  const maxGap = Math.round(maxGapS / HOP_S);
  const idx = [];
  for (let i = 0; i < m.length; i++) if (m[i]) idx.push(i);
  if (idx.length < 2) return m;
  for (let k = 0; k + 1 < idx.length; k++) {
    const a = idx[k], b = idx[k + 1];
    if (b - a > 1 && b - a <= maxGap + 1) for (let i = a; i < b; i++) m[i] = true;
  }
  return m;
}

export function longestRun(mask) {
  let best = null, bestLen = 0, start = -1;
  for (let i = 0; i <= mask.length; i++) {
    const on = i < mask.length && mask[i];
    if (on && start < 0) start = i;
    if (!on && start >= 0) {
      if (i - start > bestLen) { bestLen = i - start; best = [start, i]; }
      start = -1;
    }
  }
  return best;
}

function refineEdge(p, idx, threshold) {
  const i = Math.min(Math.max(idx, 1), p.length - 1);
  const a = p[i - 1], b = p[i];
  if (Math.abs(b - a) < 1e-9) return i * HOP_S;
  const frac = Math.min(Math.max((threshold - a) / (b - a), 0), 1);
  return (i - 1 + frac) * HOP_S;
}

export function toInterval(prob, threshold = 0.5, maxGapS = 3.0) {
  const p = medfilt(prob);
  // Именно Array.from, а не p.map: у Float64Array map возвращает Float64Array
  // и булевы значения молча превращаются в 0/1.
  const run = longestRun(closeGaps(Array.from(p, v => v > threshold), maxGapS));
  if (!run) return null;
  const [a, b] = run;
  const start = a > 0 ? refineEdge(p, a, threshold) : 0;
  let end = b < p.length ? refineEdge(p, b, threshold) : p.length * HOP_S;
  if (end <= start) end = start + HOP_S;
  let conf = 0;
  if (b > a) { for (let i = a; i < b; i++) conf += p[i]; conf /= b - a; }
  return { start, end, confidence: conf, smoothed: p };
}
