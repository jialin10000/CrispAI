const { app, imaging, core } = require("photoshop");
const { entrypoints, storage } = require("uxp");
const fs = storage.localFileSystem;

const SERVER = "http://127.0.0.1:7788";
const PREVIEW_SIZE = 1024;

entrypoints.setup({
  panels: {
    "crispai-panel": {
      show(node) {
        if (!node || node.querySelector("#btn-open")) return;
        node.innerHTML = PANEL_HTML;
        attachStyles(node);
        setupUI(node);
      },
      hide() {}
    }
  }
});

// ── HTML template ──────────────────────────────────────────────
const PANEL_HTML = `
  <div class="panel">
    <div class="panel-logo">CrispAI</div>
    <div id="panel-status" class="panel-status">Ready</div>
    <button id="btn-open" class="btn-launch">Open CrispAI...</button>
  </div>

  <!-- No-document overlay -->
  <div id="no-doc-modal" class="modal" style="display:none;">
    <div class="nodoc-box">
      <div class="nodoc-title">No Photo Open</div>
      <div class="nodoc-msg">Open a photo in Photoshop first, or choose a file below.</div>
      <button id="btn-pick-file" class="btn btn-apply" style="width:100%;">Open Photo...</button>
      <button id="btn-nodoc-cancel" class="btn btn-cancel" style="width:100%; margin-top:6px;">Cancel</button>
    </div>
  </div>

  <!-- Main compare modal -->
  <div id="crispai-modal" class="modal" style="display:none;">
    <div class="dialog-layout">

      <div class="compare-wrap">
        <div class="compare-container" id="compare-container">
          <img id="img-original" class="img-layer img-original" />
          <div class="processed-clip" id="processed-clip">
            <img id="img-processed" class="img-layer img-processed" />
          </div>
          <div class="divider" id="divider">
            <div class="divider-handle">&#9664;&#9654;</div>
          </div>
          <div class="label label-left">Original</div>
          <div class="label label-right">CrispAI</div>
        </div>
        <div id="compare-placeholder" class="compare-placeholder">Loading preview...</div>
      </div>

      <div class="controls">
        <div class="ctrl-title">CrispAI</div>

        <div class="ctrl-section">
          <div class="ctrl-label">Noise Reduction</div>
          <div class="ctrl-row">
            <input type="range" id="denoise-strength" min="0" max="100" value="50" />
            <span id="denoise-val">50</span>
          </div>
        </div>

        <div class="ctrl-section">
          <div class="ctrl-label">Sharpening</div>
          <div class="ctrl-row">
            <input type="range" id="sharpen-strength" min="0" max="100" value="50" />
            <span id="sharpen-val">50</span>
          </div>
          <select id="sharpen-mode">
            <option value="auto">Auto</option>
            <option value="motion_blur">Motion Blur (Shake)</option>
            <option value="focus_blur">Focus Blur (Soft)</option>
          </select>
        </div>

        <button id="btn-preview" class="btn btn-secondary">Update Preview</button>
        <div id="ctrl-status" class="ctrl-status">Ready</div>

        <div class="ctrl-buttons">
          <button id="btn-cancel" class="btn btn-cancel">Cancel</button>
          <button id="btn-apply" class="btn btn-apply">Apply to PS</button>
        </div>
        <div class="ctrl-note">Result added as new layer</div>
      </div>
    </div>
  </div>
`;

// ── CSS ────────────────────────────────────────────────────────
function attachStyles(node) {
  const style = document.createElement("style");
  style.textContent = `
    * { box-sizing: border-box; margin: 0; padding: 0; }
    .panel { padding: 14px; display: flex; flex-direction: column; gap: 8px; }
    .panel-logo { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: 1px; }
    .panel-status { font-size: 10px; color: #888; }
    .btn-launch { padding: 8px; background: #2d5a8e; border: 1px solid #3a72b0;
      border-radius: 4px; color: #fff; font-size: 12px; font-weight: 600; cursor: pointer; }
    .btn-launch:hover { background: #3468a3; }

    .modal { position: fixed; inset: 0; z-index: 999; }
    .dialog-layout { display: flex; width: 100%; height: 100%; background: #1a1a1a; }

    .nodoc-box { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
      background: #2a2a2a; border: 1px solid #444; border-radius: 8px;
      padding: 24px; width: 240px; display: flex; flex-direction: column; gap: 12px; }
    .nodoc-title { font-size: 15px; font-weight: 700; color: #fff; }
    .nodoc-msg { font-size: 11px; color: #999; line-height: 1.5; }

    .compare-wrap { flex: 1; background: #111; display: flex; align-items: center;
      justify-content: center; position: relative; overflow: hidden; }
    .compare-container { position: relative; display: none; cursor: ew-resize; user-select: none; }
    .compare-container.ready { display: block; }
    .img-layer { display: block; }
    .img-original { position: relative; z-index: 1; }
    .processed-clip { position: absolute; top: 0; left: 0; width: 50%; height: 100%;
      overflow: hidden; z-index: 2; }
    .img-processed { position: absolute; top: 0; left: 0; }
    .divider { position: absolute; top: 0; left: 50%; transform: translateX(-50%);
      width: 3px; height: 100%; background: #fff; z-index: 10; cursor: ew-resize; }
    .divider-handle { position: absolute; top: 50%; left: 50%;
      transform: translate(-50%,-50%); background: #fff; color: #333;
      border-radius: 50%; width: 32px; height: 32px; display: flex;
      align-items: center; justify-content: center; font-size: 10px; }
    .label { position: absolute; top: 12px; background: rgba(0,0,0,0.55);
      color: #fff; font-size: 11px; padding: 3px 8px; border-radius: 3px;
      z-index: 5; pointer-events: none; }
    .label-left { left: 12px; } .label-right { right: 12px; }
    .compare-placeholder { color: #555; font-size: 13px; }

    .controls { width: 220px; flex-shrink: 0; background: #252525; padding: 20px 16px;
      display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
    .ctrl-title { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: 1px; }
    .ctrl-section { display: flex; flex-direction: column; gap: 6px; }
    .ctrl-label { font-size: 10px; text-transform: uppercase; color: #888;
      letter-spacing: 0.5px; font-weight: 600; }
    .ctrl-row { display: flex; align-items: center; gap: 8px; }
    input[type="range"] { flex: 1; accent-color: #5aabff; }
    .ctrl-row span { width: 24px; text-align: right; color: #ccc; font-size: 11px; }
    select { width: 100%; background: #333; border: 1px solid #444; color: #ccc;
      padding: 4px 6px; border-radius: 3px; font-size: 11px; }
    .ctrl-status { font-size: 11px; color: #888; min-height: 16px; text-align: center; }
    .ctrl-status.working { color: #5aabff; } .ctrl-status.done { color: #4caf7d; }
    .ctrl-status.error { color: #ff6b6b; }
    .ctrl-buttons { display: flex; gap: 8px; margin-top: auto; }
    .btn { flex: 1; padding: 9px 4px; border-radius: 4px; font-size: 12px;
      font-weight: 600; cursor: pointer; border: 1px solid transparent; }
    .btn-secondary { background: #333; border-color: #444; color: #ccc; }
    .btn-cancel { background: #333; border-color: #555; color: #aaa; }
    .btn-apply { background: #2d5a8e; border-color: #3a72b0; color: #fff; }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .ctrl-note { font-size: 10px; color: #555; text-align: center; }
  `;
  node.appendChild(style);
}

// ── Wire up all UI events ──────────────────────────────────────
function setupUI(node) {
  const btnOpen       = node.querySelector("#btn-open");
  const panelStatus   = node.querySelector("#panel-status");
  const noDocModal    = node.querySelector("#no-doc-modal");
  const btnPickFile   = node.querySelector("#btn-pick-file");
  const btnNodocCancel= node.querySelector("#btn-nodoc-cancel");
  const modal         = node.querySelector("#crispai-modal");
  const btnPreview    = node.querySelector("#btn-preview");
  const btnApply      = node.querySelector("#btn-apply");
  const btnCancel     = node.querySelector("#btn-cancel");
  const ctrlStatus    = node.querySelector("#ctrl-status");
  const imgOriginal   = node.querySelector("#img-original");
  const imgProcessed  = node.querySelector("#img-processed");
  const container     = node.querySelector("#compare-container");
  const processedClip = node.querySelector("#processed-clip");
  const divider       = node.querySelector("#divider");
  const placeholder   = node.querySelector("#compare-placeholder");
  const denoiseSlider = node.querySelector("#denoise-strength");
  const denoiseVal    = node.querySelector("#denoise-val");
  const sharpenSlider = node.querySelector("#sharpen-strength");
  const sharpenVal    = node.querySelector("#sharpen-val");
  const sharpenMode   = node.querySelector("#sharpen-mode");

  denoiseSlider.addEventListener("input", () => denoiseVal.textContent = denoiseSlider.value);
  sharpenSlider.addEventListener("input", () => sharpenVal.textContent = sharpenSlider.value);

  // ── Open button ──
  btnOpen.addEventListener("click", async () => {
    if (!app.activeDocument) {
      noDocModal.style.display = "block";
      return;
    }
    await openMainModal();
  });

  // ── No-doc modal: pick a file ──
  btnPickFile.addEventListener("click", async () => {
    try {
      const file = await fs.getFileForOpening({
        allowMultiple: false,
        types: ["jpg", "jpeg", "png", "tif", "tiff", "psd", "psb"],
      });
      if (!file) return;
      panelStatus.textContent = "Opening...";
      noDocModal.style.display = "none";
      await app.open(file);
      await openMainModal();
    } catch (e) {
      panelStatus.textContent = "Could not open file";
    }
  });

  btnNodocCancel.addEventListener("click", () => {
    noDocModal.style.display = "none";
  });

  // ── Main modal ──
  btnCancel.addEventListener("click", () => { modal.style.display = "none"; });

  btnPreview.addEventListener("click", async () => {
    const doc = app.activeDocument;
    if (!doc) return;
    await loadPreview(doc);
  });

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
      setTimeout(() => { modal.style.display = "none"; }, 800);
    } catch (e) {
      setCtrlStatus("Error: " + e.message, "error");
      setButtonsDisabled(false);
    }
  });

  // ── Divider drag ──
  let dragging = false;
  divider.addEventListener("mousedown", (e) => { dragging = true; e.preventDefault(); });
  document.addEventListener("mouseup", () => { dragging = false; });
  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    const pct = (Math.max(0, Math.min(e.clientX - rect.left, rect.width)) / rect.width) * 100;
    processedClip.style.width = pct + "%";
    divider.style.left = pct + "%";
  });
  container.addEventListener("click", (e) => {
    const rect = container.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    processedClip.style.width = pct + "%";
    divider.style.left = pct + "%";
  });

  // ── Helpers ──
  function setCtrlStatus(text, cls = "") {
    ctrlStatus.textContent = text;
    ctrlStatus.className = "ctrl-status" + (cls ? " " + cls : "");
  }
  function setButtonsDisabled(v) {
    [btnPreview, btnApply, btnCancel].forEach(b => b.disabled = v);
  }
  function resetDivider() {
    processedClip.style.width = "50%";
    divider.style.left = "50%";
  }

  async function openMainModal() {
    panelStatus.textContent = "Loading...";
    try {
      await checkServer();
      modal.style.display = "flex";
      await loadPreview(app.activeDocument);
      panelStatus.textContent = "Ready";
    } catch (e) {
      panelStatus.textContent = "Server not running";
      modal.style.display = "none";
    }
  }

  async function loadPreview(doc) {
    placeholder.style.display = "block";
    container.classList.remove("ready");
    setCtrlStatus("Generating preview...", "working");
    try {
      const { b64, width, height } = await getLayerPixels(doc, PREVIEW_SIZE);
      imgOriginal.src = `data:image/png;base64,${b64}`;
      imgOriginal.style.width  = width + "px";
      imgOriginal.style.height = height + "px";
      setCtrlStatus("Running AI on preview...", "working");
      const result = await callBackend("enhance", b64);
      imgProcessed.src = `data:image/png;base64,${result.image}`;
      imgProcessed.style.width  = width + "px";
      imgProcessed.style.height = height + "px";
      placeholder.style.display = "none";
      container.classList.add("ready");
      resetDivider();
      setCtrlStatus("Preview ready — drag divider to compare", "done");
    } catch (e) {
      setCtrlStatus("Preview failed: " + e.message, "error");
    }
  }
}

// ── PS pixel helpers ───────────────────────────────────────────
async function getLayerPixels(doc, maxSize = null) {
  const layer = doc.activeLayers[0];
  const bounds = layer.bounds;
  const w = Math.round(bounds.right - bounds.left);
  const h = Math.round(bounds.bottom - bounds.top);
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
      left: Math.round(bounds.left), top: Math.round(bounds.top),
      right: Math.round(bounds.right), bottom: Math.round(bounds.bottom),
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

async function placeResultAsLayer(doc, b64, width, height, name) {
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

// ── Network helpers ────────────────────────────────────────────
async function checkServer() {
  const r = await fetch(`${SERVER}/health`);
  if (!r.ok) throw new Error("Server not responding");
}

async function callBackend(action, b64Image) {
  const root = document.querySelector("#crispai-modal");
  const body = {
    image: b64Image,
    denoise_strength: (root ? root.querySelector("#denoise-strength").value : 50) / 100,
    sharpen_strength: (root ? root.querySelector("#sharpen-strength").value : 50) / 100,
    sharpen_mode: root ? root.querySelector("#sharpen-mode").value : "auto",
  };
  const r = await fetch(`${SERVER}/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) { const e = await r.json(); throw new Error(e.error || `HTTP ${r.status}`); }
  return r.json();
}
