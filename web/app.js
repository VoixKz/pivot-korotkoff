/* Демонстрация модели в браузере: файл -> признаки -> ONNX -> границы. */
import * as dsp from "./dsp.js";
import * as post from "./postprocess.js";

const $ = id => document.getElementById(id);
let session = null, meta = null, current = null, audioEl = null, raf = null;

/* ------------------------------------------------------------------ модель */
async function loadModel() {
  if (session) return session;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = 1;
  meta = await (await fetch("./model_meta.json")).json();
  session = await ort.InferenceSession.create("./model.onnx",
    { executionProviders: ["wasm"] });
  $("modelinfo").textContent =
    `модель загружена · ${meta.n_features} признака на кадр · обучена на ${meta.n_train} записях · IoU ${meta.iou.toFixed(3)}`;
  return session;
}

/* ------------------------------------------------------------------- аудио */
/* WAV разбирается вручную: у записей этого прибора заголовок объявляет 60 с,
 * а данных меньше. Браузерный decodeAudioData доверяет заголовку и может
 * дорисовать тишину, из-за которой границы уедут. */
function parseWav(buf) {
  const v = new DataView(buf);
  const tag = o => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1),
                                       v.getUint8(o + 2), v.getUint8(o + 3));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
  let pos = 12, fmt = null;
  while (pos + 8 <= buf.byteLength) {
    const id = tag(pos), size = v.getUint32(pos + 4, true), body = pos + 8;
    if (id === "fmt ") {
      fmt = { format: v.getUint16(body, true), channels: v.getUint16(body + 2, true),
              sr: v.getUint32(body + 4, true), bits: v.getUint16(body + 14, true) };
    } else if (id === "data") {
      if (!fmt || fmt.bits !== 16 || fmt.format !== 1) return null;
      const avail = buf.byteLength - body;
      const nBytes = Math.min(size, avail) & ~1;
      const raw = new Int16Array(buf, body, Math.floor(nBytes / 2));
      const ch = fmt.channels;
      const n = Math.floor(raw.length / ch);
      const x = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < ch; c++) s += raw[i * ch + c];
        x[i] = s / ch;
      }
      return { x, sr: fmt.sr, truncated: size > avail };
    }
    pos = body + size + (size & 1);
  }
  return null;
}

async function resampleTo(x, srIn, srOut) {
  if (srIn === srOut) return Float32Array.from(x);
  const ctx = new OfflineAudioContext(1, Math.max(1, Math.ceil(x.length * srOut / srIn)), srOut);
  const buf = ctx.createBuffer(1, x.length, srIn);
  buf.copyToChannel(Float32Array.from(x), 0);
  const src = ctx.createBufferSource();
  src.buffer = buf; src.connect(ctx.destination); src.start();
  return (await ctx.startRendering()).getChannelData(0);
}

async function loadAudio(file) {
  const buf = await file.arrayBuffer();
  const wav = parseWav(buf);
  if (wav) {
    const x = await resampleTo(wav.x, wav.sr, dsp.C.SR);
    return { x, srIn: wav.sr, truncated: wav.truncated, via: "разбор WAV" };
  }
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(buf.slice(0));
  ctx.close();
  const mono = decoded.numberOfChannels > 1
    ? Float32Array.from({ length: decoded.length }, (_, i) => {
        let s = 0;
        for (let c = 0; c < decoded.numberOfChannels; c++) s += decoded.getChannelData(c)[i];
        return s / decoded.numberOfChannels;
      })
    : decoded.getChannelData(0);
  const x = await resampleTo(mono, decoded.sampleRate, dsp.C.SR);
  return { x, srIn: decoded.sampleRate, truncated: false, via: "декодер браузера" };
}

/* Слышимая версия: тоны Короткова лежат в 20-200 Гц и на динамиках почти
 * не слышны. Полоса, понижение до 4 кГц, нормировка по перцентилю с мягким
 * ограничением — по пику один хлопок делает все удары неслышными. */
function playableWav(x) {
  const y = dsp.sosfiltfilt(x);
  const n = Math.floor(y.length / 2);
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = y[2 * i];
  const sorted = Array.from(d, Math.abs).sort((a, b) => a - b);
  let scale = sorted[Math.floor(sorted.length * 0.9)] || sorted[sorted.length - 1] || 1;
  const pcm = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    pcm[i] = Math.max(0, Math.min(255, Math.round(Math.tanh(d[i] / scale) * 0.9 * 127) + 128));
  }
  const hdr = new ArrayBuffer(44), v = new DataView(hdr);
  const put = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  put(0, "RIFF"); v.setUint32(4, 36 + n, true); put(8, "WAVE");
  put(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, 4000, true); v.setUint32(28, 4000, true);
  v.setUint16(32, 1, true); v.setUint16(34, 8, true);
  put(36, "data"); v.setUint32(40, n, true);
  return URL.createObjectURL(new Blob([hdr, pcm], { type: "audio/wav" }));
}

/* ----------------------------------------------------------------- анализ */
async function analyse(file) {
  $("status").textContent = "читаю файл…";
  $("result").hidden = true;
  const t0 = performance.now();

  const { x, srIn, truncated, via } = await loadAudio(file);
  if (x.length < dsp.C.SR) throw new Error("запись короче секунды");

  $("status").textContent = "считаю признаки…";
  await new Promise(r => setTimeout(r, 0));
  const f = dsp.features(x);
  if (f.T < 20) throw new Error("слишком мало кадров");

  $("status").textContent = "прогоняю модель…";
  await loadModel();
  const tensor = new ort.Tensor("float32", f.data, [1, f.T, dsp.C.N_BANDS + 2]);
  const logits = (await session.run({ features: tensor })).logits.data;
  const prob = post.sigmoid(logits);
  const iv = post.toInterval(prob);

  const dur = x.length / dsp.C.SR;
  current = { x, dur, prob, iv, env: f.env, per: f.per, name: file.name,
              srIn, truncated, via, ms: performance.now() - t0 };

  if (audioEl) { audioEl.pause(); URL.revokeObjectURL(audioEl.src); }
  audioEl = new Audio(playableWav(x));
  audioEl.preservesPitch = false;
  audioEl.mozPreservesPitch = false;
  audioEl.webkitPreservesPitch = false;
  audioEl.playbackRate = +($("rate").value);
  audioEl.addEventListener("timeupdate", updateTime);
  audioEl.addEventListener("ended", () => { stopLoop(); $("play").textContent = "▶ Слушать"; draw(); });

  render();
}

function render() {
  const c = current;
  $("result").hidden = false;
  $("fname").textContent = c.name;
  $("fmeta").textContent =
    `${c.dur.toFixed(1)} с · исходно ${c.srIn} Гц · ${c.via} · обработано за ${Math.round(c.ms)} мс`;
  if (c.iv) {
    $("t0").textContent = c.iv.start.toFixed(2);
    $("t1").textContent = c.iv.end.toFixed(2);
    $("tlen").textContent = (c.iv.end - c.iv.start).toFixed(2);
    $("tconf").textContent = c.iv.confidence.toFixed(2);
    $("verdict").textContent = "";
  } else {
    $("t0").textContent = $("t1").textContent = $("tlen").textContent = $("tconf").textContent = "—";
    $("verdict").textContent = "Модель не нашла интервала ударов в этой записи.";
  }
  $("warn").hidden = !c.truncated;
  resize();
}

/* ---------------------------------------------------------------- отрисовка */
const cv = () => $("cv"), ctx = () => cv().getContext("2d");
const W = () => cv().width / devicePixelRatio;
const H = () => cv().height / devicePixelRatio;
const t2x = t => (t / current.dur) * W();

function resize() {
  if (!current) return;
  const el = cv(), r = el.getBoundingClientRect();
  el.width = Math.max(1, r.width * devicePixelRatio);
  el.height = Math.max(1, r.height * devicePixelRatio);
  ctx().setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  draw();
}
addEventListener("resize", resize);

function series(g, arr, top, height, colour, fill, normalise = true) {
  if (!arr || !arr.length) return;
  let mx = 1;
  if (normalise) { mx = 0; for (const v of arr) mx = Math.max(mx, v); mx = mx || 1; }
  g.strokeStyle = colour; g.lineWidth = 1.3; g.beginPath();
  arr.forEach((v, i) => {
    const x = (i / (arr.length - 1 || 1)) * W();
    const y = top + height - (v / mx) * height;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
  if (fill) {
    g.lineTo(W(), top + height); g.lineTo(0, top + height); g.closePath();
    g.fillStyle = fill; g.fill();
  }
}

function draw() {
  if (!current) return;
  const g = ctx(), c = current;
  g.clearRect(0, 0, W(), H());

  if (c.iv) {
    g.fillStyle = "rgba(92,201,138,.13)";
    g.fillRect(t2x(c.iv.start), 0, t2x(c.iv.end) - t2x(c.iv.start), H());
  }

  g.font = "10px system-ui, sans-serif";
  const step = c.dur > 25 ? 5 : c.dur > 10 ? 2 : 1;
  for (let s = 0; s <= c.dur; s += step) {
    const x = t2x(s);
    g.strokeStyle = "#20252b"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H()); g.stroke();
    g.fillStyle = "#5c656f"; g.fillText(s + " с", x + 3, H() - 4);
  }

  series(g, c.env, 8, H() * 0.42, "#7fd67f", "rgba(127,214,127,.10)");
  series(g, Array.from(c.prob), H() * 0.50, H() * 0.30, "#ffb648", "rgba(255,182,72,.12)", false);
  series(g, c.per, H() * 0.84, H() * 0.13, "#68b8e8", "rgba(104,184,232,.10)", false);

  // линия порога 0.5 для кривой вероятности
  const yThr = H() * 0.50 + H() * 0.30 * 0.5;
  g.strokeStyle = "rgba(255,182,72,.35)"; g.setLineDash([4, 4]); g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, yThr); g.lineTo(W(), yThr); g.stroke(); g.setLineDash([]);

  if (c.iv) {
    [["start", c.iv.start], ["end", c.iv.end]].forEach(([k, t]) => {
      const x = t2x(t);
      g.strokeStyle = "#5cc98a"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H()); g.stroke();
      g.fillStyle = "#5cc98a"; g.font = "12px system-ui, sans-serif";
      const lbl = t.toFixed(2) + " с";
      g.fillText(lbl, k === "start" ? x + 6 : x - g.measureText(lbl).width - 6, 14);
    });
  }

  if (audioEl) {
    const x = t2x(Math.min(audioEl.currentTime, c.dur));
    g.strokeStyle = "#f5f5f5"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H()); g.stroke();
    g.fillStyle = "#f5f5f5";
    g.beginPath(); g.moveTo(x - 5, 0); g.lineTo(x + 5, 0); g.lineTo(x, 8); g.closePath(); g.fill();
  }
  updateTime();
}

/* ------------------------------------------------------------------ плеер */
const fmt = t => {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0").replace(".", ",")}`;
};
function updateTime() {
  if (!current) return;
  const t = audioEl ? audioEl.currentTime : 0;
  $("time").textContent = `${fmt(t)} / ${fmt(current.dur)}`;
  $("fill").style.width = (100 * Math.min(t / current.dur, 1)) + "%";
}
function loop() { updateTime(); draw(); raf = requestAnimationFrame(loop); }
function stopLoop() { if (raf) cancelAnimationFrame(raf); raf = null; }

$("play").onclick = () => {
  if (!audioEl) return;
  if (audioEl.paused) {
    if (audioEl.currentTime >= current.dur - 0.05) audioEl.currentTime = 0;
    audioEl.play().then(() => { $("play").textContent = "⏸ Пауза"; loop(); }).catch(() => {});
  } else { audioEl.pause(); stopLoop(); $("play").textContent = "▶ Слушать"; draw(); }
};
$("rate").onchange = e => { if (audioEl) audioEl.playbackRate = +e.target.value; };
$("seek").onclick = e => {
  if (!audioEl) return;
  const r = $("seek").getBoundingClientRect();
  audioEl.currentTime = Math.max(0, Math.min(current.dur - 0.01,
    ((e.clientX - r.left) / r.width) * current.dur));
  updateTime(); draw();
};
document.addEventListener("click", e => {
  if (e.target === cv() && audioEl) {
    const r = cv().getBoundingClientRect();
    audioEl.currentTime = Math.max(0, Math.min(current.dur - 0.01,
      ((e.clientX - r.left) / r.width) * current.dur));
    updateTime(); draw();
  }
});

/* ------------------------------------------------------------------- ввод */
async function handle(file) {
  try {
    await analyse(file);
    $("status").textContent = "";
  } catch (err) {
    $("status").textContent = "Не получилось: " + err.message;
    $("result").hidden = true;
    console.error(err);
  }
}
$("file").onchange = e => { if (e.target.files[0]) handle(e.target.files[0]); };
const drop = $("drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add("over");
}));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove("over");
}));
drop.addEventListener("drop", e => { if (e.dataTransfer.files[0]) handle(e.dataTransfer.files[0]); });

$("demo").onclick = async () => {
  $("status").textContent = "загружаю пример…";
  try {
    const r = await fetch("./sample.wav");
    if (!r.ok) throw new Error("пример не найден");
    await handle(new File([await r.blob()], "пример — recording (629).wav"));
  } catch (err) { $("status").textContent = "Пример недоступен: " + err.message; }
};

loadModel().catch(err => {
  $("modelinfo").textContent = "модель не загрузилась: " + err.message;
});
