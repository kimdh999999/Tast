import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

// 모델 캐시 (2회차부터 즉시 실행)
let cachedLabels = null;
let cachedSession = null;
let cachedLandmarker = null;
let activeCleanup = null;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (window.ort) { resolve(); return; }
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

function toWristRelative(vector) {
  const out = Float32Array.from(vector);
  for (let hand = 0; hand < 2; hand++) {
    const start = hand * 63;
    let present = false;
    for (let i = start; i < start + 63; i++) { if (out[i] !== 0) { present = true; break; } }
    if (!present) continue;
    const wx = out[start], wy = out[start + 1], wz = out[start + 2];
    for (let k = 0; k < 21; k++) {
      const i = start + k * 3;
      out[i] -= wx; out[i + 1] -= wy; out[i + 2] -= wz;
    }
  }
  return out;
}

function toScaleNormalised(vector) {
  const out = Float32Array.from(vector);
  for (let hand = 0; hand < 2; hand++) {
    const start = hand * 63;
    let present = false;
    for (let i = start; i < start + 63; i++) { if (out[i] !== 0) { present = true; break; } }
    if (!present) continue;
    let size = 0;
    for (let k = 0; k < 21; k++) {
      const i = start + k * 3;
      const d = Math.hypot(out[i], out[i + 1], out[i + 2]);
      if (d > size) size = d;
    }
    const divisor = Math.max(size, 1e-8);
    for (let i = start; i < start + 63; i++) out[i] /= divisor;
  }
  return out;
}

function transform(vector) {
  return toScaleNormalised(toWristRelative(vector));
}

function handMask(v) {
  let left = false, right = false;
  for (let i = 0; i < 63; i++) { if (v[i] !== 0) { left = true; break; } }
  for (let i = 63; i < 126; i++) { if (v[i] !== 0) { right = true; break; } }
  return (left ? 1 : 0) | (right ? 2 : 0);
}

class RollingBuffer {
  constructor(length = 12, minFrames = 4) {
    this.length = length; this.minFrames = minFrames;
    this.frames = []; this.mask = null;
  }
  push(vector) {
    if (!vector) return;
    const m = handMask(vector);
    if (m !== this.mask) { this.frames.length = 0; this.mask = m; }
    this.frames.push(vector);
    if (this.frames.length > this.length) this.frames.shift();
  }
  clear() { this.frames.length = 0; this.mask = null; }
  get ready() { return this.frames.length >= this.minFrames; }
  value() {
    if (!this.ready) return null;
    const n = this.frames.length;
    const out = new Float32Array(126);
    const col = new Array(n);
    const mid = n >> 1;
    const even = (n % 2 === 0);
    for (let d = 0; d < 126; d++) {
      for (let f = 0; f < n; f++) col[f] = this.frames[f][d];
      col.sort((a, b) => a - b);
      out[d] = even ? (col[mid - 1] + col[mid]) / 2 : col[mid];
    }
    return out;
  }
}

class LetterMerger {
  constructor({ tau = 0.35, minFrames = 8, minHoldTime = 0.35, idleFramesToForget = 12 } = {}) {
    this.tau = tau; this.minFrames = minFrames;
    this.minHoldTime = minHoldTime; this.idleFramesToForget = idleFramesToForget;
    this.idleFrames = 0; this.candidate = null; this.lastCommitted = null;
  }
  update(label, confidence, now = performance.now() / 1000) {
    if (!label || confidence < this.tau) {
      this.idleFrames++;
      this.candidate = null;
      if (this.idleFrames >= this.idleFramesToForget) this.lastCommitted = null;
      return null;
    }
    this.idleFrames = 0;
    if (!this.candidate || this.candidate.label !== label) {
      this.candidate = { label, frames: 0, sum: 0, startedAt: now, committed: false };
    }
    const c = this.candidate;
    c.frames++; c.sum += confidence;
    if (c.committed) return null;
    if (!(c.frames >= this.minFrames && (now - c.startedAt) >= this.minHoldTime && (c.sum / c.frames) >= this.tau)) {
      return null;
    }
    c.committed = true;
    if (label === this.lastCommitted) return null;
    this.lastCommitted = label;
    return label;
  }
}

function toRawVector(result) {
  if (!result?.landmarks?.length) return null;
  const vector = new Float32Array(126);
  const pts = result.landmarks[0];
  const start = 63;
  for (let k = 0; k < 21; k++) {
    const p = pts[k];
    const at = start + k * 3;
    vector[at] = p.x; vector[at + 1] = p.y; vector[at + 2] = p.z;
  }
  return vector;
}

function drawSkeleton(ctx, landmarks, w, h) {
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "#00FF00";
  ctx.fillStyle = "#FF0000";
  for (const pts of landmarks) {
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      if (!pts[a] || !pts[b]) continue;
      ctx.moveTo(pts[a].x * w, pts[a].y * h);
      ctx.lineTo(pts[b].x * w, pts[b].y * h);
    }
    ctx.stroke();
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

async function ensureModels(modelBase) {
  if (cachedLabels && cachedSession && cachedLandmarker) return;

  await loadScript("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js");
  window.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  window.ort.env.wasm.numThreads = 1;

  const [labelsData, sess, fileset] = await Promise.all([
    fetch(`${modelBase}/models/labels.json`).then((r) => r.json()),
    window.ort.InferenceSession.create(`${modelBase}/models/asl_svm.onnx`, { executionProviders: ["wasm"] }),
    FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm")
  ]);

  const options = {
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.35,
    minHandPresenceConfidence: 0.35,
    minTrackingConfidence: 0.35
  };

  let lm = null;
  try {
    lm = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${modelBase}/models/hand_landmarker.task`, delegate: "CPU" },
      ...options
    });
  } catch (e) {
    lm = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "CPU"
      },
      ...options
    });
  }

  cachedLabels = labelsData;
  cachedSession = sess;
  cachedLandmarker = lm;
}

export async function startAslInColab(uid, modelBase = "https://kimdh999999.github.io/Tast") {
  if (typeof activeCleanup === "function") {
    try { activeCleanup(); } catch (e) {}
    activeCleanup = null;
  }

  const root = document.getElementById(uid);
  if (!root) return;

  const video = root.querySelector(".webcam");
  const canvas = root.querySelector(".canvas");
  const ctx = canvas.getContext("2d");
  const currentEl = root.querySelector(".current-letter");
  const textBoxEl = root.querySelector(".text-box");
  const state = window._aslBridge[uid];

  let stopped = false;
  let stream = null;
  let rafId = null;
  let isBusy = false;
  let lastTimestamp = -1;
  let noHandFrames = 0;

  const buffer = new RollingBuffer(12, 4);
  const merger = new LetterMerger();

  function updateText() {
    textBoxEl.textContent = state.text;
  }

  function cleanup() {
    stopped = true;
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }

  activeCleanup = cleanup;
  state.cleanup = cleanup;
  state.updateText = updateText;

  async function loop(now) {
    if (stopped) return;
    if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
      rafId = requestAnimationFrame(loop);
      return;
    }

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const w = canvas.width, h = canvas.height;

    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    let ts = Math.round(now);
    if (ts <= lastTimestamp) ts = lastTimestamp + 1;
    lastTimestamp = ts;

    let result = null;
    if (cachedLandmarker) {
      try { result = cachedLandmarker.detectForVideo(canvas, ts); } catch (e) {}
    }

    if (result?.landmarks?.length) {
      drawSkeleton(ctx, result.landmarks, w, h);
    }

    const vector = toRawVector(result);
    if (vector) { noHandFrames = 0; buffer.push(vector); }
    else { noHandFrames++; if (noHandFrames >= 5) buffer.clear(); }

    if (!isBusy && cachedSession) {
      isBusy = true;
      try {
        const aggregated = buffer.value();
        let predictedLabel = null, confidence = 0;

        if (aggregated && vector) {
          const feat = transform(aggregated);
          const tensor = new window.ort.Tensor("float32", feat, [1, 126]);
          const output = await cachedSession.run({ input: tensor });
          const probs = output.probabilities.data;
          let best = 0;
          for (let i = 1; i < probs.length; i++) { if (probs[i] > probs[best]) best = i; }
          predictedLabel = cachedLabels[best] ?? null;
          confidence = probs[best];
        }

        currentEl.textContent = predictedLabel
          ? predictedLabel
          : (result?.landmarks?.length ? "동작 분석 중..." : "-");

        const committed = merger.update(predictedLabel, confidence, now / 1000);
        if (committed) {
          state.text += committed;
          updateText();
        }
      } catch (err) {
        console.error("추론 오류:", err);
      } finally {
        isBusy = false;
      }
    }

    rafId = requestAnimationFrame(loop);
  }

  currentEl.textContent = "카메라 연결 중...";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
  } catch (camErr) {
    currentEl.textContent = `카메라 오류 (${camErr.name || "권한 확인"})`;
    console.error("카메라 오류:", camErr);
    return;
  }

  rafId = requestAnimationFrame(loop);

  currentEl.textContent = "AI 로딩 중...";
  try {
    await ensureModels(modelBase);
    currentEl.textContent = "-";
  } catch (err) {
    currentEl.textContent = "모델 로딩 오류";
    console.error("모델 로딩 실패:", err);
  }
}
