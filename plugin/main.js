const { app, imaging, core } = require("photoshop");
const { entrypoints, storage, shell } = require("uxp");
const fs = storage.localFileSystem;

const SERVER = "http://localhost:7788";

// ── Panel UI ───────────────────────────────────────────────────
entrypoints.setup({
  panels: {
    "crispai-panel": {
      show(node) {
        if (node.querySelector("#btn-open")) return;
        node.style.cssText = "width:100%;height:100%;display:block;";

        const style = document.createElement("style");
        style.textContent = `
          * { box-sizing: border-box; margin: 0; padding: 0; }
          .panel { padding: 16px; display: flex; flex-direction: column; gap: 12px; }
          .logo  { font-size: 18px; font-weight: 700; color: #fff; letter-spacing: 1.5px; }
          .sub   { font-size: 11px; color: #444; line-height: 1.5; }
          .status { font-size: 11px; color: #666; min-height: 16px; }
          .status.working { color: #4a9eff; }
          .status.error   { color: #ff6b6b; }
          .status.done    { color: #4caf7d; }
          .btn {
            padding: 10px; background: #1d5bbf;
            border: 1px solid #2469d6; border-radius: 6px;
            color: #fff; font-size: 13px; font-weight: 600;
            cursor: pointer; text-align: center;
          }
          .btn:hover { background: #2166cc; }
          .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        `;
        node.appendChild(style);

        node.innerHTML += `
          <div class="panel">
            <div class="logo">CrispAI</div>
            <div class="sub">AI-powered noise reduction &amp; sharpening</div>
            <div class="status" id="status">Ready</div>
            <button class="btn" id="btn-open">Open in CrispAI…</button>
          </div>
        `;

        node.querySelector("#btn-open").addEventListener("click", () => {
          const btn    = node.querySelector("#btn-open");
          const status = node.querySelector("#status");
          openCrispAI(btn, status);
        });
      },
      hide() {}
    }
  }
});

// ── Main flow ──────────────────────────────────────────────────
async function openCrispAI(btn, status) {
  btn.disabled = true;

  const setStatus = (text, cls = "") => {
    status.textContent = text;
    status.className = "status" + (cls ? " " + cls : "");
  };

  try {
    // 1. Check server
    setStatus("Connecting to server…", "working");
    const health = await fetch(`${SERVER}/health`).catch(() => null);
    if (!health || !health.ok) {
      // Try to auto-start server
      setStatus("Starting server…", "working");
      try {
        const pluginFolder = await fs.getPluginFolder();
        const bat = await pluginFolder.getEntry("start-server.bat");
        await shell.openPath(bat.nativePath);
        let ok = false;
        for (let i = 0; i < 10; i++) {
          await sleep(1000);
          const r = await fetch(`${SERVER}/health`).catch(() => null);
          if (r && r.ok) { ok = true; break; }
        }
        if (!ok) { setStatus("Server not running — start it manually.", "error"); btn.disabled = false; return; }
      } catch (_) {
        setStatus("Server not running.", "error"); btn.disabled = false; return;
      }
    }

    // 2. Check / open document
    let doc = app.activeDocument;
    if (!doc) {
      setStatus("Choose a photo…");
      const file = await fs.getFileForOpening({
        allowMultiple: false,
        types: ["jpg","jpeg","png","tif","tiff","psd","psb"],
      });
      if (!file) { setStatus("Ready"); btn.disabled = false; return; }
      setStatus("Opening…", "working");
      await core.executeAsModal(async () => { await app.open(file); },
        { commandName: "CrispAI: open file" });
      doc = app.activeDocument;
    }

    // 3. Read pixels (requires modal scope)
    setStatus("Reading pixels…", "working");
    let b64, width, height;
    await core.executeAsModal(async () => {
      const r = await getLayerPixels(doc);
      b64 = r.b64; width = r.width; height = r.height;
    }, { commandName: "CrispAI: read pixels" });

    // 4. Upload to server → get session URL
    setStatus("Uploading…", "working");
    const createResp = await fetch(`${SERVER}/session/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64, format: "rgba8", width, height }),
    });
    const { session_id, url } = await createResp.json();

    // 5. Open CrispAI in browser
    await shell.openExternal(url);
    setStatus("Waiting for you to apply in CrispAI…");

    // 6. Poll until user clicks Apply or Cancel
    const result = await pollResult(session_id, setStatus);
    if (!result) { setStatus("Cancelled"); btn.disabled = false; return; }

    // 7. Place result as new PS layer
    setStatus("Placing result…", "working");
    await placeResult(doc, result);
    setStatus("Done!", "done");
    setTimeout(() => setStatus("Ready"), 3000);

  } catch (e) {
    setStatus("Error: " + (e.message || String(e)), "error");
  }

  btn.disabled = false;
}

// ── Polling ────────────────────────────────────────────────────
async function pollResult(sid, setStatus) {
  for (let i = 0; i < 1200; i++) {     // up to 20 min
    await sleep(1000);
    const r = await fetch(`${SERVER}/session/${sid}/result`).catch(() => null);
    if (!r) continue;
    const data = await r.json();
    if (data.status === "ready")     return data;
    if (data.status === "cancelled") return null;
    if (data.status === "not_found") return null;
    const m = String(Math.floor(i / 60)).padStart(2, "0");
    const s = String(i % 60).padStart(2, "0");
    setStatus(`Waiting… (${m}:${s})`);
  }
  return null;
}

// ── Place result layer ─────────────────────────────────────────
async function placeResult(doc, result) {
  const bytes = base64ToUint8(result.raw_rgba);
  const { width, height } = result;
  await core.executeAsModal(async () => {
    const layer = await doc.createPixelLayer({ name: "CrispAI" });
    await imaging.putPixels({
      documentID: doc.id,
      layerID:    layer.id,
      componentSize: 8,
      colorProfile:  "sRGB IEC61966-2.1",
      colorSpace:    "RGB",
      imageData: imaging.createImageDataFromBuffer(bytes.buffer, {
        width, height, components: 4,
        colorProfile: "sRGB IEC61966-2.1",
      }),
    });
  }, { commandName: "CrispAI: place result" });
}

// ── PS pixel read ──────────────────────────────────────────────
async function getLayerPixels(doc) {
  const layer  = doc.activeLayers[0];
  const bounds = layer.bounds;
  const w = Math.round(bounds.right  - bounds.left);
  const h = Math.round(bounds.bottom - bounds.top);

  const pixelData = await imaging.getPixels({
    documentID:   doc.id,
    layerID:      layer.id,
    componentSize: 8,
    colorProfile:  "sRGB IEC61966-2.1",
    colorSpace:    "RGB",
    bounds: {
      left:   Math.round(bounds.left),
      top:    Math.round(bounds.top),
      right:  Math.round(bounds.right),
      bottom: Math.round(bounds.bottom),
    },
  });

  const buffer = await pixelData.imageData.getData();
  return { b64: uint8ToBase64(new Uint8Array(buffer)), width: w, height: h };
}

// ── Encode / decode helpers ────────────────────────────────────
function uint8ToBase64(bytes) {
  let binary = "";
  const chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk)
    binary += String.fromCharCode.apply(null, bytes.subarray(i, Math.min(i + chunk, bytes.length)));
  return btoa(binary);
}

function base64ToUint8(b64) {
  const binary = atob(b64);
  const bytes  = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
