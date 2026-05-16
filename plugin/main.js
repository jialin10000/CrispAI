const { app, imaging, core } = require("photoshop");
const { entrypoints, storage, shell } = require("uxp");
const fs = storage.localFileSystem;

const SERVER = "http://localhost:7788"; // local backend (ASUS RTX 4070)
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
  <!-- Default state -->
  <div id="view-main" class="panel">
    <div class="panel-logo">CrispAI</div>
    <div id="panel-status" class="panel-status">Ready</div>
    <button id="btn-open" class="btn-launch">Open CrispAI...</button>
  </div>

  <!-- Server not running state -->
  <div id="view-noserver" class="panel" style="display:none;">
    <div class="panel-logo">CrispAI</div>
    <div class="nodoc-msg">Backend server is not running.</div>
    <div class="server-cmd">cd backend<br/>python server.py</div>
    <button id="btn-retry" class="btn-launch">Retry</button>
    <button id="btn-noserver-cancel" class="btn-back">&#8592; Back</button>
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
  // Make the node fill the panel and allow absolute children
  node.style.cssText = "width:100%; height:100%; position:relative; display:block;";

  const style = document.createElement("style");
  style.textContent = `
    * { box-sizing: border-box; margin: 0; padding: 0; }
    .panel { padding: 14px; display: flex; flex-direction: column; gap: 8px; width: 100%; }
    .panel-logo { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: 1px; }
    .panel-status { font-size: 10px; color: #888; }
    .btn-launch { padding: 8px; background: #2d5a8e; border: 1px solid #3a72b0;
      border-radius: 4px; color: #fff; font-size: 12px; font-weight: 600; cursor: pointer; }
    .btn-launch:hover { background: #3468a3; }

    .modal { width: 100%; }
    .dialog-layout { display: flex; flex-direction: column; width: 100%; background: #1a1a1a; }

    .nodoc-msg { font-size: 11px; color: #999; line-height: 1.5; }
    .server-cmd { font-size: 10px; color: #5aabff; background: #1a1a1a;
      border-radius: 3px; padding: 6px 8px; font-family: monospace; line-height: 1.6; }
    .btn-back { background: none; border: none; color: #666; font-size: 11px;
      cursor: pointer; text-align: left; padding: 2px 0; }

    .compare-wrap { width: 100%; min-height: 200px; background: #111; display: flex;
      align-items: center; justify-content: center; position: relative; overflow: hidden; }
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
    .compare-placeholder { color: #aaa; font-size: 11px; padding: 12px;
      white-space: pre-wrap; word-break: break-all; overflow-y: auto;
      max-height: 300px; text-align: left; width: 100%; }

    .controls { width: 100%; background: #252525; padding: 12px 14px;
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
  const btnOpen           = node.querySelector("#btn-open");
  const panelStatus       = node.querySelector("#panel-status");
  const viewMain          = node.querySelector("#view-main");
  const viewNoserver      = node.querySelector("#view-noserver");
  const btnRetry          = node.querySelector("#btn-retry");
  const btnNoserverCancel = node.querySelector("#btn-noserver-cancel");
  const modal          = node.querySelector("#crispai-modal");

  function showView(name) {
    viewMain.style.display     = name === "main"     ? "flex" : "none";
    viewNoserver.style.display = name === "noserver" ? "flex" : "none";
    modal.style.display        = name === "modal"    ? "flex" : "none";
  }
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

  // ── Open button: if no doc, go straight to file picker ──
  btnOpen.addEventListener("click", async () => {
    const doc = app.activeDocument;
    if (!doc) { await pickAndOpenFile(); return; }
    await openMainModal();
  });

  async function pickAndOpenFile() {
    try {
      panelStatus.textContent = "Choose a photo...";
      const file = await fs.getFileForOpening({
        allowMultiple: false,
        types: ["jpg", "jpeg", "png", "tif", "tiff", "psd", "psb"],
      });
      if (!file) { panelStatus.textContent = "Ready"; return; }
      panelStatus.textContent = "Opening...";
      await core.executeAsModal(async () => {
        await app.open(file);
      }, { commandName: "CrispAI: Open Photo" });
      await openMainModal();
    } catch (e) {
      panelStatus.textContent = "Error: " + (e.message || e);
    }
  }

  // ── No-server: retry ──
  btnRetry.addEventListener("click", () => openMainModal());
  btnNoserverCancel.addEventListener("click", () => showView("main"));

  // ── Main modal ──
  btnCancel.addEventListener("click", () => { showView("main"); });

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
      const result = await callBackend("enhance", b64, {
        denoiseStrength: +denoiseSlider.value,
        sharpenStrength: +sharpenSlider.value,
        sharpenMode: sharpenMode.value,
      });
      setCtrlStatus("Placing result in Photoshop...", "working");
      await placeResultAsLayer(doc, result.image, width, height, "CrispAI");
      setCtrlStatus("Done!", "done");
      setTimeout(() => { showView("main"); }, 800);
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
    showView("main");
    try {
      await checkServer();
    } catch (e) {
      // Server not running — try to auto-start it
      panelStatus.textContent = "Starting server...";
      try {
        const pluginFolder = await fs.getPluginFolder();
        const batFile = await pluginFolder.getEntry("start-server.bat");
        await shell.openPath(batFile.nativePath);
        // Wait up to 8 seconds for server to come up
        let started = false;
        for (let i = 0; i < 8; i++) {
          await new Promise(r => setTimeout(r, 1000));
          try { await checkServer(); started = true; break; } catch (_) {}
        }
        if (!started) { showView("noserver"); panelStatus.textContent = "Ready"; return; }
      } catch (e2) {
        showView("noserver"); panelStatus.textContent = "Ready"; return;
      }
    }
    showView("modal");
    await loadPreview(app.activeDocument);
    panelStatus.textContent = "Ready";
  }

  async function loadPreview(doc) {
    placeholder.style.display = "block";
    placeholder.textContent = "Loading preview...";
    container.classList.remove("ready");
    setCtrlStatus("Generating preview...", "working");
    try {
      // getPixels requires modal scope
      let b64, width, height;
      await core.executeAsModal(async () => {
        const result = await getLayerPixels(doc, PREVIEW_SIZE);
        b64 = result.b64; width = result.width; height = result.height;
      }, { commandName: "CrispAI: read pixels" });

      imgOriginal.src = `data:image/png;base64,${b64}`;
      imgOriginal.style.width  = width + "px";
      imgOriginal.style.height = height + "px";
      setCtrlStatus("Running AI on preview...", "working");
      const result = await callBackend("enhance", b64, {
        denoiseStrength: +denoiseSlider.value,
        sharpenStrength: +sharpenSlider.value,
        sharpenMode: sharpenMode.value,
      });
      imgProcessed.src = `data:image/png;base64,${result.image}`;
      imgProcessed.style.width  = width + "px";
      imgProcessed.style.height = height + "px";
      placeholder.style.display = "none";
      container.classList.add("ready");
      resetDivider();
      setCtrlStatus("Preview ready — drag divider to compare", "done");
    } catch (e) {
      placeholder.textContent = "Error: " + (e.message || String(e));
      placeholder.style.display = "block";
      setCtrlStatus("Preview failed", "error");
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
  const canvas = document.createElement("canvas");
  canvas.width = pw;
  canvas.height = ph;
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(pw, ph);
  imageData.data.set(new Uint8ClampedArray(buffer));
  ctx.putImageData(imageData, 0, 0);
  const dataUrl = canvas.toDataURL("image/png");
  const b64 = dataUrl.split(",")[1];
  return { b64, width: pw, height: ph };
}

async function placeResultAsLayer(doc, b64, width, height, name) {
  const img = new Image();
  img.src = `data:image/png;base64,${b64}`;
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
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

async function callBackend(action, b64Image, params = {}) {
  const body = {
    image: b64Image,
    denoise_strength: (params.denoiseStrength ?? 50) / 100,
    sharpen_strength: (params.sharpenStrength ?? 50) / 100,
    sharpen_mode: params.sharpenMode ?? "auto",
  };
  const r = await fetch(`${SERVER}/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) { const e = await r.json(); throw new Error(e.error || `HTTP ${r.status}`); }
  return r.json();
}
