# UXP Photoshop Plugin Panel Rendering — Key Findings

> Applies to: Adobe Photoshop 27.6.0 / UXP 7.x

## Problem

After loading the plugin, the panel showed only the title bar ("CrispAI") but the
body was completely blank. DevTools showed `<head></head><body></body>`. No console
errors. Reloading made no difference.

---

## Root Causes (all four must be fixed)

### 1. Missing root-level `main.js`

UXP bootstrap looks for `main.js` at the **plugin root** (same directory as
`manifest.json`) during startup. Without it, the entire plugin fails silently and
no panel content is ever rendered.

```
plugin/
  manifest.json
  main.js        ← REQUIRED — UXP bootstrap entry point
  src/
    ...
```

Error when missing:
```
Uncaught Error: Cannot resolve module: main.js.
  at e.exports.loadDefaultMainFile (uxp://uxp-internal/pluginmanager_scripts.js)
```

---

### 2. `manifestVersion: 4` + `apiVersion: 2` does not work

`manifestVersion: 4` silently ignores the `apiVersion` field, causing PS to warn
"Plugin is using deprecated apiVersion of 1".

**Fix**: Use `manifestVersion: 5` with `host` as an array:

```json
{
  "manifestVersion": 5,
  "host": [{ "app": "PS", "minVersion": "22.0.0" }]
}
```

---

### 3. `"main": "index.html"` in the panel entrypoint does NOT render HTML

In UXP 7.x, setting `"main": "src/index.html"` (or any HTML path) in the panel
entrypoint **does not** load that file as panel content. The body stays empty
regardless of whether the HTML is in `src/` or at the plugin root.

**Do not do this:**
```json
{
  "type": "panel",
  "id": "my-panel",
  "main": "index.html"   ← ignored in UXP 7.x
}
```

---

### 4. `show({ node })` — wrong destructuring, `node` is `undefined`

UXP passes the panel's root DOM element **directly** as the argument to `show()`,
not wrapped in an object.

```js
// ❌ Wrong — node will be undefined
show({ node } = {}) { node.innerHTML = "..."; }

// ✅ Correct
show(node) {
  if (!node) return;
  node.innerHTML = "...";
}
```

This produces the error:
```
TypeError: Cannot set properties of undefined (setting 'innerHTML')
  at e.exports.show (plugin/main.js)
  Entrypoint: create
```

---

## Working Solution

### manifest.json

```json
{
  "id": "com.example.plugin",
  "name": "My Plugin",
  "version": "1.0.0",
  "manifestVersion": 5,
  "host": [{ "app": "PS", "minVersion": "22.0.0" }],
  "entrypoints": [
    {
      "type": "panel",
      "id": "my-panel",
      "label": { "default": "My Plugin" },
      "minimumSize": { "width": 200, "height": 120 },
      "defaultSize": { "width": 240, "height": 160 }
    }
  ],
  "requiredPermissions": {
    "network": { "domains": ["localhost"] },
    "localFileSystem": "request"
  }
}
```

Note: **no `"main"` field inside the panel entrypoint**.

### plugin/main.js

```js
const { app } = require("photoshop");
const { entrypoints } = require("uxp");

entrypoints.setup({
  panels: {
    "my-panel": {
      show(node) {
        // Guard against multiple calls
        if (!node || node.querySelector("#root")) return;

        // Inject HTML
        node.innerHTML = `<div id="root"><p>Hello from UXP!</p></div>`;

        // Inject CSS
        const style = document.createElement("style");
        style.textContent = `* { box-sizing: border-box; } p { color: white; }`;
        node.appendChild(style);

        // Bind events using node.querySelector (not document.querySelector)
        node.querySelector("p").addEventListener("click", () => {
          console.log("clicked");
        });
      },
      hide() {}
    }
  }
});
```

### Key rules

| Rule | Why |
|------|-----|
| `show(node)` not `show({ node })` | UXP passes the DOM node directly |
| `node.querySelector()` not `document.querySelector()` | Scoped to panel |
| Inject `<style>` via `document.createElement` | `<link>` rel resolution unreliable |
| Use `<div style="position:fixed;inset:0">` for modals | UXP does not support `<dialog>` |
| `require("photoshop")` at top level is fine | Works in UXP module scope |
| Guard `show()` with an already-rendered check | `show()` is called each time panel becomes visible |

---

## File Picker (open photo from within plugin)

```js
const { storage } = require("uxp");
const fs = storage.localFileSystem;

const file = await fs.getFileForOpening({
  allowMultiple: false,
  types: ["jpg", "jpeg", "png", "tif", "tiff", "psd"],
});
if (file) {
  await app.open(file);
}
```
