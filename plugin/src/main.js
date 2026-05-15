import { app, imaging, core } from "photoshop";

const SERVER = "http://127.0.0.1:7788";

// --- UI bindings ---

const denoiseSlider = document.getElementById("denoise-strength");
const denoiseVal    = document.getElementById("denoise-val");
const sharpenSlider = document.getElementById("sharpen-strength");
const sharpenVal    = document.getElementById("sharpen-val");
const sharpenMode   = document.getElementById("sharpen-mode");
const statusEl      = document.getElementById("status");

denoiseSlider.addEventListener("input", () => denoiseVal.textContent = denoiseSlider.value);
sharpenSlider.addEventListener("input", () => sharpenVal.textContent = sharpenSlider.value);

document.getElementById("btn-denoise").addEventListener("click", () => run("denoise"));
document.getElementById("btn-sharpen").addEventListener("click", () => run("sharpen"));
document.getElementById("btn-enhance").addEventListener("click", () => run("enhance"));

// --- Core logic ---

function setStatus(text, cls = "idle") {
  statusEl.textContent = text;
  statusEl.className = `status ${cls}`;
}

function setButtonsDisabled(disabled) {
  document.querySelectorAll(".btn").forEach(b => b.disabled = disabled);
}

async function run(action) {
  const doc = app.activeDocument;
  if (!doc) {
    setStatus("No document open", "error");
    return;
  }

  setStatus("Exporting layer...", "working");
  setButtonsDisabled(true);

  try {
    // Export active layer as PNG to temp file
    const tempPath = await exportActiveLayer(doc);

    // Send to backend
    setStatus("Processing with AI...", "working");
    const b64 = await fileToBase64(tempPath);
    const result = await callBackend(action, b64);

    // Import result back as new layer
    setStatus("Importing result...", "working");
    await importAsNewLayer(doc, result.image, `CrispAI ${action}`);

    setStatus("Done!", "done");
    setTimeout(() => setStatus("Ready"), 3000);
  } catch (err) {
    setStatus(`Error: ${err.message}`, "error");
    console.error(err);
  } finally {
    setButtonsDisabled(false);
  }
}

async function callBackend(action, b64Image) {
  const body = { image: b64Image };

  if (action === "denoise" || action === "enhance") {
    body.denoise_strength = denoiseSlider.value / 100;
  }
  if (action === "sharpen" || action === "enhance") {
    body.sharpen_strength = sharpenSlider.value / 100;
    body.sharpen_mode = sharpenMode.value;
  }

  const endpoint = action === "enhance" ? "enhance"
    : action === "denoise" ? "denoise"
    : "sharpen";

  const resp = await fetch(`${SERVER}/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}

async function exportActiveLayer(doc) {
  // Use PS imaging API to get pixel data from active layer
  const layer = doc.activeLayers[0];
  const pixels = await imaging.getPixels({
    documentID: doc.id,
    layerID: layer.id,
    componentSize: 8,
    colorProfile: "sRGB IEC61966-2.1",
    colorSpace: "RGB",
  });
  return pixels; // returns ImageData-like object
}

async function fileToBase64(pixels) {
  // Convert PS pixel data to base64 PNG
  const buffer = await pixels.imageData.getData();
  const arr = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < arr.byteLength; i++) {
    binary += String.fromCharCode(arr[i]);
  }
  return btoa(binary);
}

async function importAsNewLayer(doc, b64Image, layerName) {
  await core.executeAsModal(async () => {
    const layer = await doc.createPixelLayer({ name: layerName });
    // Paste the processed image data back
    // Full implementation requires writing to temp file and placing
    // This is a placeholder — will flesh out in next iteration
    console.log("Result layer created:", layerName);
  }, { commandName: "CrispAI: import result" });
}
