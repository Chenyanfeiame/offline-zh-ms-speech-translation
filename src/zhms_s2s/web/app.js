const $ = (selector) => document.querySelector(selector);
const languages = {
  zh: { name: "中文", target: "ms", sourceLabel: "中文识别", targetLabel: "马来语直译" },
  ms: { name: "Bahasa Melayu", target: "zh", sourceLabel: "马来语识别", targetLabel: "中文直译" },
};
let sourceLanguage = "auto";
let audioBlob = null;
let recorder = null;
let timer = null;
let recordingStarted = 0;
let animation = null;
let activeRequestSequence = 0;
let activeRequestController = null;
let currentWorkflow = null;
let currentDelivery = null;
let lastTranslatedSource = "";
const recordButton = $("#record-button");
const translateButton = $("#translate-button");
const fileInput = $("#file-input");
const dropzone = $("#dropzone");
const canvas = $("#waveform");
const context2d = canvas.getContext("2d");
const recordButtonText = $("#record-button-text");
const warmupPanel = $("#warmup-panel");
const warmupNote = $("#warmup-note");
const warmupButtonLabel = $("#warmup-button-label");

function modelsAreWarm(health) {
  const components = health?.components;
  const asrModels = Object.values(components?.asr?.models || {});
  const asrWarm = asrModels.length
    ? asrModels.every((model) => model.loaded && model.inference_warmed)
      && components?.asr?.language_id?.inference_warmed
    : components?.asr?.loaded && components?.asr?.inference_warmed;
  const translationModels = Object.values(components?.translation?.models || {});
  const defaultZhVoice = components?.tts?.voices?.zh?.default;
  const zhOptions = components?.tts?.voices?.zh?.options || {};
  return Boolean(
    asrWarm
    && translationModels.length
    && translationModels.every((model) => model.loaded)
    && defaultZhVoice
    && zhOptions[defaultZhVoice]?.loaded
  );
}

function showWarmupReady(seconds = null) {
  warmupPanel.className = "warmup-panel ready";
  warmupButtonLabel.textContent = "预热完成";
  warmupNote.textContent = seconds == null
    ? "模型已经在内存中，可以直接开始录音。"
    : `用时 ${seconds.toFixed(1)} 秒。模型已进入内存，可以开始录音。`;
  $("#warmup-button").disabled = true;
}

function updateLanguages() {
  const automatic = sourceLanguage === "auto";
  const target = automatic ? null : languages[sourceLanguage].target;
  $("#source-language").value = sourceLanguage;
  $("#source-mode").textContent = automatic ? "自动语言识别" : sourceLanguage === "zh" ? "已锁定中文" : "已锁定马来语";
  $("#target-language").textContent = automatic ? "自动反向" : languages[target].name;
  $("#target-code").textContent = automatic ? "中文 ↔ 马来语" : target === "zh" ? "输出中文" : "输出马来语";
  $("#swap-language small").textContent = automatic ? "手动" : "切换";
  updateVoiceOptions(target);
}

function updateVoiceOptions(target) {
  const select = $("#voice-select");
  const options = target === "zh"
    ? [["xiao_ya", "Xiao Ya · 女声（默认）"], ["huayan", "Huayan · 男声"]]
    : target === "ms"
      ? [["mms-zlm", "MMS Malay · 单说话人"]]
      : [["", "自动匹配：中文女声 / 马来语 MMS"]];
  select.innerHTML = options.map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
}

function chooseBlob(blob, name) {
  invalidateActiveRequest();
  clearResults();
  audioBlob = blob;
  $("#file-name").textContent = name;
  translateButton.disabled = false;
  translateButton.querySelector("b").textContent = "音频已就绪";
  hideError();
}

function drawIdle() {
  context2d.clearRect(0, 0, canvas.width, canvas.height);
  context2d.strokeStyle = "rgba(225,48,108,.15)";
  context2d.beginPath();
  for (let x = 0; x < canvas.width; x += 6) {
    const y = canvas.height / 2 + Math.sin(x * 0.04) * 5 * Math.sin(x * 0.006);
    x === 0 ? context2d.moveTo(x, y) : context2d.lineTo(x, y);
  }
  context2d.stroke();
}

function encodeWav(chunks, sampleRate) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + length * 2, true); text(8, "WAVE"); text(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, "data"); view.setUint32(40, length * 2, true);
  let offset = 44;
  chunks.forEach((chunk) => chunk.forEach((sample) => {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
    offset += 2;
  }));
  return new Blob([buffer], { type: "audio/wav" });
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("当前浏览器不支持麦克风录音，请改用上传音频。");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = (event) => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  source.connect(analyser); source.connect(processor); processor.connect(audioContext.destination);
  recorder = { stream, audioContext, source, analyser, processor, chunks };
  recordButton.classList.add("recording");
  recordButton.setAttribute("aria-label", "停止录音");
  recordButtonText.textContent = "停止录音";
  $("#record-label").textContent = "正在录音 · 再点一次结束";
  $("#record-time").textContent = "00:00";
  recordingStarted = Date.now();
  timer = setInterval(() => {
    const seconds = Math.floor((Date.now() - recordingStarted) / 1000);
    $("#record-time").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    if (seconds >= 300) stopRecording();
  }, 250);
  visualize(analyser);
}

function visualize(analyser) {
  const data = new Uint8Array(analyser.frequencyBinCount);
  const paint = () => {
    analyser.getByteTimeDomainData(data);
    context2d.clearRect(0, 0, canvas.width, canvas.height);
    context2d.strokeStyle = "rgba(225,48,108,.68)"; context2d.lineWidth = 2; context2d.beginPath();
    data.forEach((value, index) => {
      const x = index / (data.length - 1) * canvas.width;
      const y = value / 255 * canvas.height;
      index ? context2d.lineTo(x, y) : context2d.moveTo(x, y);
    });
    context2d.stroke(); animation = requestAnimationFrame(paint);
  };
  paint();
}

async function stopRecording() {
  if (!recorder) return;
  clearInterval(timer); cancelAnimationFrame(animation);
  const { stream, audioContext, source, processor, chunks } = recorder;
  processor.disconnect(); source.disconnect(); stream.getTracks().forEach((track) => track.stop());
  const sampleRate = audioContext.sampleRate; await audioContext.close(); recorder = null;
  recordButton.classList.remove("recording"); recordButton.setAttribute("aria-label", "开始录音");
  recordButtonText.textContent = "重新录音";
  $("#record-label").textContent = "录音完成，可重新录制";
  const seconds = (Date.now() - recordingStarted) / 1000;
  if (seconds < .25) { showError("录音太短，请至少说一秒钟。"); drawIdle(); return; }
  chooseBlob(encodeWav(chunks, sampleRate), `recording-${Date.now()}.wav`); drawIdle();
}

async function toggleRecording() {
  hideError();
  try { recorder ? await stopRecording() : await startRecording(); } catch (error) { showError(error.message); }
}

function showError(message) { $("#error-message").textContent = message; $("#error").classList.remove("hidden"); }
function hideError() { $("#error").classList.add("hidden"); }

function invalidateActiveRequest() {
  activeRequestSequence += 1;
  if (activeRequestController) activeRequestController.abort();
  activeRequestController = null;
}

function clearResults() {
  $("#results").classList.add("hidden");
  $("#asr-text").value = "";
  $("#translated-text").textContent = "";
  $("#request-id").textContent = "";
  $("#timings").innerHTML = "";
  $("#entity-rescue-summary").classList.add("hidden");
  $("#entity-rescue-summary p").textContent = "";
  $("#entity-rescue-summary small").textContent = "";
  $("#translation-card").classList.remove("hidden");
  $("#audio-card").classList.remove("hidden");
  $("#regenerate-button").disabled = true;
  $("#edit-hint").textContent = "如果识别不准，可以直接修改文本后重新生成译文和语音。";
  const audio = $("#output-audio");
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  currentWorkflow = null;
  currentDelivery = null;
  lastTranslatedSource = "";
}

function newRequestId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (char) => {
    const random = Math.floor(Math.random() * 16);
    const value = char === "x" ? random : (random & 0x3) | 0x8;
    return value.toString(16);
  });
}

async function translate() {
  if (!audioBlob) return;
  invalidateActiveRequest();
  const requestSequence = activeRequestSequence;
  const clientRequestId = newRequestId();
  const controller = new AbortController();
  activeRequestController = controller;
  clearResults();
  hideError(); translateButton.disabled = true; translateButton.querySelector("span").textContent = "本地模型处理中…";
  const form = new FormData();
  form.append("file", audioBlob, audioBlob.name || "recording.wav");
  form.append("client_request_id", clientRequestId);
  if (sourceLanguage !== "auto") {
    form.append("source_language", sourceLanguage);
    form.append("target_language", languages[sourceLanguage].target);
  }
  if ($("#voice-select").value) form.append("voice_id", $("#voice-select").value);
  const entityContext = $("#entity-context").value.trim();
  if (entityContext) form.append("user_context", entityContext);
  try {
    const response = await fetch("/speech-translate", { method: "POST", body: form, signal: controller.signal });
    const data = await response.json();
    if (requestSequence !== activeRequestSequence) return;
    if (!response.ok) throw new Error(data?.error?.message || `请求失败（HTTP ${response.status}）`);
    if (data.request_id !== clientRequestId) throw new Error("服务返回的 request_id 与当前请求不一致，结果已拒绝显示。");
    renderResult(data);
  } catch (error) {
    if (requestSequence !== activeRequestSequence || error.name === "AbortError") return;
    clearResults();
    showError(error.message || "无法连接本地服务。");
  } finally {
    if (requestSequence === activeRequestSequence) {
      activeRequestController = null;
      translateButton.disabled = false;
      translateButton.querySelector("span").textContent = "开始离线翻译";
    }
  }
}

function renderResult(data) {
  const confidence = data.language_confidence == null ? "手动指定" : `自动判定 · ${(data.language_confidence * 100).toFixed(1)}%`;
  $("#asr-language").textContent = `${languages[data.source_language].sourceLabel} · ${confidence}`;
  $("#asr-text").value = data.enhanced_asr_text || data.raw_asr_text;
  currentWorkflow = {
    parentRequestId: data.request_id,
    rawAsrText: data.raw_asr_text,
    sourceLanguage: data.source_language,
    targetLanguage: data.target_language,
    initialTimings: data.timings,
  };
  lastTranslatedSource = data.actual_mt_input || data.raw_asr_text;
  currentDelivery = data;
  const entitySummary = $("#entity-rescue-summary");
  const entityNames = [...new Set((data.detected_entities || []).map((item) => item.canonical_name))];
  if (entityNames.length) {
    entitySummary.querySelector("p").textContent = entityNames.join(" · ");
    entitySummary.querySelector("small").textContent = data.asr_passes > 1
      ? "已通过否定词、数字与实体注入安全检查"
      : "词典别名或已识别实体";
    entitySummary.classList.remove("hidden");
  } else {
    entitySummary.classList.add("hidden");
  }
  $("#regenerate-button").disabled = true;
  $("#edit-hint").textContent = "如果识别不准，可以直接修改文本后重新生成译文和语音。";
  renderDelivery(data, false);
  $("#results").classList.remove("hidden");
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  if (data.language_confidence != null && data.language_confidence < 0.7) {
    showError("语言判断置信度较低。请检查转写；如有误，可在来源语言中手动选择后重试。");
  }
}

function renderDelivery(data, regenerated) {
  $("#translation-card").classList.remove("hidden");
  $("#audio-card").classList.remove("hidden");
  $("#mt-language").textContent = languages[data.source_language].targetLabel;
  $("#translated-text").textContent = data.translated_text;
  $("#request-id").textContent = regenerated
    ? `ID ${data.request_id.slice(0, 12)} · edited`
    : `ID ${data.request_id.slice(0, 12)}`;
  $("#output-audio").src = `${data.output_audio_url}?v=${Date.now()}`;
  const timings = data.timings;
  const timingRows = regenerated
    ? [["ASR", currentWorkflow?.initialTimings?.asr_ms || 0], ["MT", timings.mt_ms], ["TTS", timings.tts_ms], ["REGENERATE", timings.total_ms]]
    : [["DECODE", timings.audio_decode_ms], ["ASR", timings.asr_ms], ["MT", timings.mt_ms], ["TTS", timings.tts_ms], ["TOTAL", timings.total_ms]];
  $("#timings").innerHTML = timingRows
    .map(([label, value]) => `<span>${label} ${Math.round(value)} ms</span>`).join("")
    + (!regenerated && timings.rtf != null ? `<span>RTF ${Number(timings.rtf).toFixed(2)}</span>` : "")
    + `<span>VOICE ${data.voice_id}</span>`;
  $("#result-state").textContent = regenerated ? "已按修改文本重新生成" : "翻译完成";
  $("#result-title").textContent = regenerated ? "新译文和语音已就绪" : "刚刚说了什么？";
}

function clearStaleDelivery() {
  $("#translation-card").classList.add("hidden");
  $("#audio-card").classList.add("hidden");
  $("#translated-text").textContent = "";
  const audio = $("#output-audio");
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  $("#result-state").textContent = "识别文本已修改";
  $("#result-title").textContent = "请重新生成译文和语音";
  $("#timings").innerHTML = "";
}

function handleAsrEdit() {
  if (!currentWorkflow) return;
  const edited = $("#asr-text").value.trim();
  const changed = Boolean(edited) && edited !== lastTranslatedSource.trim();
  $("#regenerate-button").disabled = !changed;
  if (changed) {
    clearStaleDelivery();
    $("#edit-hint").textContent = "当前译文和音频已过期。请点击下方按钮，MT 将仅使用修改后的文本。";
  } else if (edited && currentDelivery) {
    renderDelivery(currentDelivery, currentDelivery.parent_request_id != null);
    $("#edit-hint").textContent = "当前译文与识别文本一致。";
  }
}

async function regenerateFromEditedAsr() {
  if (!currentWorkflow) return;
  const editedText = $("#asr-text").value.trim();
  if (!editedText || editedText === lastTranslatedSource.trim()) return;
  invalidateActiveRequest();
  const requestSequence = activeRequestSequence;
  const requestId = newRequestId();
  const controller = new AbortController();
  activeRequestController = controller;
  const button = $("#regenerate-button");
  button.disabled = true;
  button.textContent = "正在重新翻译并生成语音…";
  hideError();
  try {
    const response = await fetch("/edited-speech-translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        request_id: requestId,
        parent_request_id: currentWorkflow.parentRequestId,
        raw_asr_text: currentWorkflow.rawAsrText,
        edited_asr_text: editedText,
        source_language: currentWorkflow.sourceLanguage,
        target_language: currentWorkflow.targetLanguage,
        voice_id: $("#voice-select").value || null,
      }),
    });
    const data = await response.json();
    if (requestSequence !== activeRequestSequence) return;
    if (!response.ok) throw new Error(data?.error?.message || `请求失败（HTTP ${response.status}）`);
    if (data.request_id !== requestId || data.parent_request_id !== currentWorkflow.parentRequestId) {
      throw new Error("request_id 与当前编辑流程不一致，结果已拒绝显示。");
    }
    if (data.actual_mt_input !== editedText) {
      throw new Error("MT 实际输入与文本框内容不一致，结果已拒绝显示。");
    }
    lastTranslatedSource = editedText;
    currentDelivery = data;
    renderDelivery(data, true);
    $("#edit-hint").textContent = "已按当前修改文本重新生成译文和语音。";
  } catch (error) {
    if (requestSequence !== activeRequestSequence || error.name === "AbortError") return;
    clearStaleDelivery();
    showError(error.message || "无法重新生成译文和语音。");
    button.disabled = false;
  } finally {
    if (requestSequence === activeRequestSequence) {
      activeRequestController = null;
      button.textContent = "按修改后的文本重新生成";
    }
  }
}

async function checkHealth() {
  const node = $("#system-status");
  if (location.protocol === "file:") {
    node.className = "status";
    node.innerHTML = "<span></span>界面预览 · 启动服务后可用";
    $("#warmup-button").disabled = true;
    warmupNote.textContent = "请先启动本地服务，并通过 http://127.0.0.1:8000 打开页面。";
    return;
  }
  try {
    const response = await fetch("/health"); const data = await response.json();
    node.className = `status ${data.ready ? "ready" : "degraded"}`;
    node.innerHTML = `<span></span>${data.ready ? "本地服务已连接" : "部分模型尚未准备"}`;
    if (modelsAreWarm(data)) showWarmupReady();
  } catch { node.className = "status degraded"; node.innerHTML = "<span></span>无法连接本地服务"; }
}

async function warmupModels() {
  const button = $("#warmup-button");
  const node = $("#system-status");
  button.disabled = true;
  warmupPanel.className = "warmup-panel loading";
  warmupButtonLabel.textContent = "正在预热…";
  warmupNote.textContent = "正在把中马 Qwen3-ASR、HY-MT 和 TTS 加载进内存，请等待。";
  node.className = "status";
  node.innerHTML = "<span></span>正在加载本地模型…";
  hideError();
  try {
    const response = await fetch("/warmup", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data?.error?.message || "模型预热失败");
    node.className = "status ready";
    node.innerHTML = `<span></span>预热完成 · ${(data.timings.total_ms / 1000).toFixed(1)} s`;
    showWarmupReady(data.timings.total_ms / 1000);
  } catch (error) {
    warmupPanel.className = "warmup-panel failed";
    warmupButtonLabel.textContent = "重新预热";
    warmupNote.textContent = "预热没有完成，请确认本地服务仍在运行后重试。";
    showError(error.message || "模型预热失败");
    button.disabled = false;
    await checkHealth();
  }
}

recordButton.addEventListener("click", async () => {
  if (recordButton.disabled) return;
  recordButton.disabled = true;
  await toggleRecording();
  recordButton.disabled = false;
});
$("#source-language").addEventListener("change", (event) => { sourceLanguage = event.target.value; updateLanguages(); });
$("#swap-language").addEventListener("click", () => { sourceLanguage = sourceLanguage === "auto" ? "zh" : languages[sourceLanguage].target; updateLanguages(); });
fileInput.addEventListener("change", () => { const file = fileInput.files[0]; if (file) chooseBlob(file, file.name); });
["dragenter", "dragover"].forEach((event) => dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((event) => dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (event) => { const file = event.dataTransfer.files[0]; if (file) chooseBlob(file, file.name); });
translateButton.addEventListener("click", translate);
$("#asr-text").addEventListener("input", handleAsrEdit);
$("#regenerate-button").addEventListener("click", regenerateFromEditedAsr);
$("#warmup-button").addEventListener("click", warmupModels);
drawIdle(); updateLanguages(); checkHealth();
