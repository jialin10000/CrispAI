const { app, imaging, core } = require("photoshop");

const SERVER = "http://127.0.0.1:7788";
const PREVIEW_SIZE = 1024; // downscale for fast preview

// ── UI elements ──
const btnOpen      = document.getElementById("btn-open");
const panelStatus  = document.getElementById("panel-status");
const dialog       = document.getElementById("crispai-dialog");
const btnPreview   = document.getElementById("btn-preview");
const btnApply     = document.getElementById("btn-apply");
const btnCancel    = document.getElementById("btn-cancel");
const ctrlStatus   = document.getElementById("ctrl-status");
const imgOriginal  = document.getElementById("img-original");
const imgProcessed = document.getElementById("img-processed");
const container    = document.getElementById("compare-container");
const processedClip = document.getElementById("processed-clip");
const divider      = document.getElementById("divider");
const placeholder  = document.getElementById("compare-placeholder");
const denoiseSlider = document.getElementById("denoise-strength");
const denoiseVal    = document.getElementById("denoise-val");
const sharpenSlider = document.getElementById("sharpen-strength");
const sharpenVal    = document.getElementById("sharpen-val");
const sharpenMode   = document.getElementById("sharpen-mode");

// ── Slider labels ──
denoiseSlider.addEventListener("input", () => denoiseVal.textContent = denoiseSlider.value);
sharpenSlider.addEventListener("input", () => sharpenVal.textContent = sharpenSlider.value);

// ── Open dialog ──
btnOpen.addEventListener("click", async () => {
  const doc = app.activeDocument;
  if (!doc) { setPanelStatus("No document open"); return; }

  setPanelStatus("Loading...");
  try {
    await checkServer();
    dialog.showModal();
    await loadPreview(doc);
    setPanelStatus("Ready");
  } catch (e) {
    setPanelStatus("Server not running");
    dialog.close();
  }
});

// ── Close ──
btnCancel.addEventListener("click", () => dialog.close());

// ── Update preview ──
btnPreview.addEventListener("click", async () => {
  const doc = app.activeDocument;
  if (!doc) return;
  await loadPreview(doc);
});

// ── Apply full resolution to PS ──
btnApply.addEventListener("click", async () => {
  const doc = app.activeDocument;
  if (!doc) return;

  setCtrlStatus("Exporting full resolution...", "working");
  setButtonsDisabled(true);

  try {
    const { b64, width, height } = await getLayerPixels(doc);
    setCtrlStatus("Processing with AI (full res)...", "working");

    const result = await callBackend("enhance", b64);

    setCtrlStatus("Placing result in Photoshop...", "working");
    await placeResultAsLayer(doc, result.image, width, height, "CrispAI");

    setCtrlStatus("Done!", "done");
    setTimeout(() => dialog.close(), 800);
  } catch (e) {
    setCtrlStatus("Error: " + e.message, "error");
    setButtonsDisabled(false);
  }
});

// ── Load preview ──
async function loadPreview(doc) {
  placeholder.style.display = "block";
  container.classList.remove("ready");
  setCtrlStatus("Generating preview...", "working");

  try {
    const { b64, width, height } = await getLayerPixels(doc, PREVIEW_SIZE);

    // Show original
    imgOriginal.src = `data:image/png;base64,${b64}`;
    imgOriginal.style.width  = width + "px";
    imgOriginal.style.height = height + "px";

    setCtrlStatus("Running AI on preview...", "working");
    const result = await callBackend("enhance", b64);

    // Show processed
    imgProcessed.src = `data:image/png;base64,${result.image}`;
    imgProcessed.style.width  = width + "px";
    imgProcessed.style.height = height + "px";

    // Show compare view
    placeholder.style.display = "none";
    container.classList.add("ready");
    resetDivider();
    setCtrlStatus("Preview ready — drag divider to compare", "done");
  } catch (e) {
    setCtrlStatus("Preview failed: " + e.message, "error");
  }
}

// ── Draggable divider ──
let dragging = false;

divider.addEventListener("mousedown", (e) => {
  dragging = true;
  e.preventDefault();
});

document.addEventListener("mouseup", () => { dragging = false; });

document.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const rect = container.getBoundingClientRect();
  const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
  const pct = (x / rect.width) * 100;
  processedClip.style.width = pct + "%";
  divider.style.left = pct + "%";
});

container.addEventListener("click", (e) => {
  const rect = container.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const pct = (x / rect.width) * 100;
  processedClip.style.width = pct + "%";
  divider.style.left = pct + "%";
});

function resetDivider() {
  processedClip.style.width = "50%";
  divider.style.left = "50%";
}

// ── Get layer pixels → base64 PNG ──
async function getLayerPixels(doc, maxSize = null) {
  const layer = doc.activeLayers[0];
  const bounds = layer.bounds;
  let w = Math.round(bounds.right - bounds.left);
  let h = Math.round(bounds.bottom - bounds.top);

  const scale = maxSize ? Math.min(1, maxSize / Math.max(w, h)) : 1;
  const pw = Math.round(w * scale);
  const ph = Math.round(h * scale);

  const pixelData = await imaging.getPixels({
    documentID: doc.id,
    layerID: layer.id,
    componentSize: 8,
    colorProfile: "sRGB IEC61966-2.1",
    colorSpace: "RGB",
    bounds: {
      left:   Math.round(bounds.left),
      top:    Math.round(bounds.top),
      right:  Math.round(bounds.right),
      bottom: Math.round(bounds.bottom),
    },
    targetSize: maxSize ? { width: pw, height: ph } : undefined,
  });

  const buffer = await pixelData.imageData.getData();
  const canvas = new OffscreenCanvas(pw, ph);
  const ctx = canvas.getContext("2d");
  ctx.putImageData(new ImageData(new Uint8ClampedArray(buffer), pw, ph), 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/png" });
  const ab = await blob.arrayBuffer();
  const bytes = new Uint8Array(ab);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);

  return { b64: btoa(binary), width: pw, height: ph };
}

// ── Place result as new layer in PS ──
async function placeResultAsLayer(doc, b64, width, height, name) {
  // Decode base64 → Uint8Array RGBA
  const binary = atob(b64);
  // We need raw RGBA from the PNG — use OffscreenCanvas to decode
  const blob = await fetch(`data:image/png;base64,${b64}`).then(r => r.blob());
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, width, height);

  await core.executeAsModal(async () => {
    const layer = await doc.createPixelLayer({ name });
    await imaging.putPixels({
      documentID: doc.id,
      layerID: layer.id,
      componentSize: 8,
      colorProfile: "sRGB IEC61966-2.1",
      colorSpace: "RGB",
      imageData: imaging.createImageDataFromBuffer(
        imageData.data.buffer,
        { width, height, components: 4, colorProfile: "sRGB IEC61966-2.1" }
      ),
    });
  }, { commandName: "CrispAI: place result" });
}

// ── Backend helpers ──
async function checkServer() {
  const r = await fetch(`${SERVER}/health`);
  if (!r.ok) throw new Error("Server not responding");
}

async function callBackend(action, b64Image) {
  const body = {
    image: b64Image,
    denoise_strength: denoiseSlider.value / 100,
    sharpen_strength: sharpenSlider.value / 100,
    sharpen_mode: sharpenMode.value,
  };
  const r = await fetch(`${SERVER}/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) { const e = await r.json(); throw new Error(e.error || `HTTP ${r.status}`); }
  return r.json();
}

// ── UI helpers ──
function setCtrlStatus(text, cls = "") {
  ctrlStatus.textContent = text;
  ctrlStatus.className = "ctrl-status" + (cls ? " " + cls : "");
}

function setPanelStatus(text) {
  panelStatus.textContent = text;
}

function setButtonsDisabled(v) {
  [btnPreview, btnApply, btnCancel].forEach(b => b.disabled = v);
}
