/* IOSCAN Scan Console — front-end controller.
 *
 * Talks to the existing FastAPI service only through its documented endpoints:
 *   GET  /api/status                 POST /api/capture/start    POST /api/capture/stop
 *   POST /api/camera/probe           POST /api/segmentation/start
 *   GET  /api/camera/preview.jpg     GET  /api/model/{preview|mesh|segmented|teeth|cloud}
 *   GET  /api/model/tooth/{1..14}
 *
 * The 3D engine (three.js) is loaded lazily. If it cannot be loaded (for example on an offline
 * workstation) the rest of the console, including capture control, keeps working.
 */

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const TARGET_FPS = 50;

/* Colours MeshSegNet writes for classes 0..14 (src/ioscan/pipeline/segment.py PALETTE). */
const SEG_PALETTE = [
  [160,160,160],[230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],[145,30,180],
  [70,240,240],[240,50,230],[210,245,60],[250,190,212],[0,128,128],[220,190,255],[170,110,40],[128,0,0],
];

const FILES = {
  mesh:      { label: 'Reconstructed surface mesh', file: 'reconstructed_mesh.ply' },
  segmented: { label: 'Segmented scan (class colours)', file: 'segmented_full_colored.ply' },
  teeth:     { label: 'Teeth only', file: 'segmented_clean_teeth.ply' },
  cloud:     { label: 'Fused point cloud', file: 'fused_cloud.ply' },
};

const S = {
  snap: null,
  connected: true,
  failures: 0,
  cam: { state: 'untested', w: 0, h: 0 },        // untested | ready | error
  lastGood: null,                                 // last non-error stage; locates where a failure happened
  clientAlert: null,                              // {title, body}
  view: 'auto',                                   // auto | preview | mesh | segmented | teeth | cloud | tooth:N
  display: 'shaded',                              // shaded | wire | points
  preset: 'iso',
  loadedKey: null, failKey: null, failAt: 0, loadToken: 0,
  origin: null, userMoved: false, needFit: false,
  fps: null, fpsPrev: null, capStart: null, capAvg: null,
  wantPreview: false, previewRunning: false, camUrl: null, camShown: false,
  starting: false, probing: false,
  sessionKey: null,
};

/* ====================================================================== utilities */
function setText(id, v) { const el = $(id); if (el && el.textContent !== String(v)) el.textContent = v; }
const fmtInt = (n) => (n || n === 0) ? Number(n).toLocaleString('en-US') : '—';
function fmtClock(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}
function basename(p) { return String(p || '').split(/[\\/]/).pop(); }
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c])); }
function store(key, val) { try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; } }

function toast(msg, kind = 'info', ms = 6000) {
  const el = document.createElement('div');
  el.className = `toast ${kind === 'err' ? 'err' : ''}`;
  el.innerHTML = `<span>${esc(msg)}</span><button aria-label="Dismiss">✕</button>`;
  el.querySelector('button').onclick = () => el.remove();
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), kind === 'err' ? Math.max(ms, 9000) : ms);
}

function formatDetail(detail) {
  if (!detail) return null;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      const field = Array.isArray(d.loc) ? d.loc.filter((x) => x !== 'body').join('.') : '';
      return field ? `${field}: ${d.msg}` : d.msg;
    }).join('; ');
  }
  return JSON.stringify(detail);
}

async function api(path, { method = 'GET', body } = {}) {
  const res = await fetch(path, {
    method, cache: 'no-store',
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  let data = null;
  if ((res.headers.get('content-type') || '').includes('json')) data = await res.json().catch(() => null);
  if (!res.ok) {
    const err = new Error(formatDetail(data && data.detail) || `${res.status} ${res.statusText}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

/* ====================================================================== theme + tabs */
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  store('ioscan.theme', t);
  $('theme-icon').innerHTML = `<use href="#${t === 'dark' ? 'i-sun' : 'i-moon'}"/>`;
  $('theme-toggle').title = t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
  V.applyTheme();
}
$('theme-toggle').addEventListener('click', () => {
  applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
});

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
  document.querySelectorAll('.pane').forEach((p) => p.classList.toggle('on', p.id === `pane-${name}`));
}
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => switchTab(t.dataset.tab)));

/* ====================================================================== setup form */
const FORM_IDS = ['camera-id', 'exposure', 'serial-port', 'session-name', 'frame-count', 'speed', 'jaw', 'mesh-method', 'calibration-file', 'models-dir', 'auto-segment'];
const PERSIST = FORM_IDS.filter((i) => i !== 'session-name');

function loadSetup() {
  try {
    const saved = JSON.parse(store('ioscan.setup') || '{}');
    for (const id of PERSIST) {
      if (!(id in saved)) continue;
      const el = $(id);
      if (el.type === 'checkbox') el.checked = !!saved[id]; else el.value = saved[id];
    }
  } catch (e) { /* ignore corrupt saved setup */ }
}
function saveSetup() {
  const o = {};
  for (const id of PERSIST) { const el = $(id); o[id] = el.type === 'checkbox' ? el.checked : el.value; }
  store('ioscan.setup', JSON.stringify(o));
}
FORM_IDS.forEach((id) => $(id).addEventListener('change', saveSetup));

function fieldError(id, msg) {
  const input = $(id), e = $(`e-${id}`);
  input.classList.toggle('invalid', !!msg);
  if (e) { e.hidden = !msg; e.textContent = msg || ''; }
  return !msg;
}

function readForm() {
  let ok = true;
  const intIn = (id, lo, hi, label) => {
    const raw = $(id).value.trim(), n = Number(raw);
    if (raw === '' || !Number.isInteger(n) || n < lo || n > hi) { fieldError(id, `${label}: whole number ${lo} to ${hi}`); ok = false; return null; }
    fieldError(id, null); return n;
  };
  const camera_id = intIn('camera-id', 0, 16, 'Camera index');
  const exposure = intIn('exposure', -13, 0, 'Exposure');
  const frames = intIn('frame-count', 20, 20000, 'Frame limit');

  const speed = Number($('speed').value);
  if (!Number.isFinite(speed) || speed <= 0.05 || speed > 50) { fieldError('speed', 'Speed: above 0.05 up to 50 mm/s'); ok = false; } else fieldError('speed', null);

  const serial_port = $('serial-port').value.trim();
  if (!serial_port || serial_port.length > 32) { fieldError('serial-port', 'Enter the ESP32 serial port, e.g. COM3'); ok = false; } else fieldError('serial-port', null);

  const session_name = $('session-name').value.trim();
  if (session_name && !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(session_name)) {
    fieldError('session-name', 'Letters, numbers, “_” and “-” only (max 64)'); ok = false;
  } else fieldError('session-name', null);

  if (!ok) { switchTab('setup'); return null; }
  const payload = {
    camera_id, serial_port, frames, exposure, speed_mm_s: speed,
    jaw: $('jaw').value, mesh_method: $('mesh-method').value,
    auto_segment: $('auto-segment').checked,
    models_dir: $('models-dir').value.trim() || null,
    calibration_file: $('calibration-file').value.trim() || null,
  };
  if (session_name) payload.session_name = session_name;
  return payload;
}

function lockForm(locked) {
  document.querySelectorAll('#pane-setup input, #pane-setup select, #pane-setup button').forEach((el) => { el.disabled = locked; });
}

/* ====================================================================== viewer (three.js) */
const V = {
  ok: false, THREE: null, renderer: null, scene: null, camera: null, controls: null,
  flip: null, group: null, object: null, geom: null, asset: null, loader: null,
  gizmoScene: null, gizmoCam: null, mats: {},

  cssColor(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888'; },

  async init() {
    const host = $('viewer');
    try {
      const THREE = await import('three');
      const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
      const { PLYLoader } = await import('three/addons/loaders/PLYLoader.js');
      this.THREE = THREE;

      const r = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      r.outputColorSpace = THREE.SRGBColorSpace;
      r.autoClear = true;
      host.insertBefore(r.domElement, host.firstChild);
      this.renderer = r;

      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 5000);
      this.camera.position.set(40, 34, 55);
      this.scene.add(this.camera);

      this.controls = new OrbitControls(this.camera, r.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.screenSpacePanning = true;
      this.controls.addEventListener('start', () => { S.userMoved = true; });

      /* Lighting: soft fill + a headlight that follows the viewer, so surface relief is always readable. */
      this.scene.add(new THREE.HemisphereLight(0xffffff, 0x70707a, 1.15));
      const head = new THREE.DirectionalLight(0xffffff, 1.9);
      head.position.set(0.4, 0.8, 1.0);
      this.camera.add(head);

      /* Scan data is in the first camera frame (x right, y down, z forward). A half-turn about X
         presents it with y up and the scanner at +Z, without any mirroring. */
      this.flip = new THREE.Group();
      this.flip.rotation.x = Math.PI;
      this.group = new THREE.Group();
      this.flip.add(this.group);
      this.scene.add(this.flip);

      this.loader = new PLYLoader();
      this.buildGizmo();
      this.applyTheme();

      new ResizeObserver(() => this.resize()).observe(host);
      this.resize();
      this.ok = true;
      $('vp-hint').hidden = false;
      const loop = () => {
        requestAnimationFrame(loop);
        if (document.hidden) return;
        this.controls.update();
        this.render();
      };
      loop();
    } catch (err) {
      console.error('3D engine unavailable:', err);
      this.ok = false;
      setText('viewer-empty-title', '3D viewer unavailable');
      setText('viewer-empty-sub', 'The 3D engine could not be loaded (offline workstation?). Capture and reconstruction still work; open exported PLY files in an external viewer. See the note in index.html about hosting three.js locally.');
      document.querySelectorAll('#seg-display button, #view-preset, #btn-fit, #btn-shot').forEach((b) => { b.disabled = true; });
    }
  },

  resize() {
    if (!this.renderer) return;
    const host = $('viewer');
    const w = Math.max(host.clientWidth, 1), h = Math.max(host.clientHeight, 1);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  },

  /* Small world-axis indicator rendered in the lower-left corner of the canvas. */
  buildGizmo() {
    const T = this.THREE;
    this.gizmoScene = new T.Scene();
    this.gizmoCam = new T.PerspectiveCamera(30, 1, 0.1, 20);
    const g = new T.Group();
    g.rotation.x = Math.PI;                       // same presentation as the model
    const axes = [['X', [1, 0, 0], 0xd64545], ['Y', [0, 1, 0], 0x2f9e5b], ['Z', [0, 0, 1], 0x3b72d9]];
    for (const [name, d, color] of axes) {
      const geo = new T.BufferGeometry().setFromPoints([new T.Vector3(0, 0, 0), new T.Vector3(...d)]);
      g.add(new T.Line(geo, new T.LineBasicMaterial({ color })));
      const c = document.createElement('canvas'); c.width = c.height = 64;
      const ctx = c.getContext('2d');
      ctx.font = '600 38px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = '#' + color.toString(16).padStart(6, '0'); ctx.fillText(name, 32, 34);
      const sp = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(c), depthTest: false, transparent: true }));
      sp.position.set(d[0] * 1.32, d[1] * 1.32, d[2] * 1.32); sp.scale.set(0.5, 0.5, 0.5);
      g.add(sp);
    }
    this.gizmoScene.add(g);
  },

  render() {
    const r = this.renderer, host = $('viewer');
    r.setScissorTest(false);
    r.setViewport(0, 0, host.clientWidth, host.clientHeight);
    r.render(this.scene, this.camera);
    const size = 84, pad = 10;
    r.autoClear = false;
    r.setScissorTest(true);
    r.setViewport(pad, pad, size, size);
    r.setScissor(pad, pad, size, size);
    r.clearDepth();
    const dir = new this.THREE.Vector3().subVectors(this.camera.position, this.controls.target).normalize().multiplyScalar(4.2);
    this.gizmoCam.position.copy(dir);
    this.gizmoCam.quaternion.copy(this.camera.quaternion);
    r.render(this.gizmoScene, this.gizmoCam);
    r.setScissorTest(false);
    r.autoClear = true;
  },

  applyTheme() {
    if (!this.ok && !this.renderer) return;
    const T = this.THREE;
    this.scene.background = new T.Color(this.cssColor('--vp-bg'));
    if (this.geom) this.rebuild();
  },

  clear() {
    if (this.object) {
      this.group.remove(this.object);
      const m = this.object.material;
      (Array.isArray(m) ? m : [m]).forEach((x) => x && x.dispose());
      this.object = null;
    }
  },

  /* Show a parsed PLY geometry according to the chosen display mode. */
  rebuild() {
    if (!this.geom) return;
    const T = this.THREE, g = this.geom;
    this.clear();
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    const hasColor = g.hasAttribute('color');
    const indexed = !!g.index;
    g.computeBoundingBox();
    const size = g.boundingBox.getSize(new T.Vector3());
    const span = Math.max(size.x, size.y, size.z, 1);

    if (!indexed || S.display === 'points') {
      const mat = new T.PointsMaterial({
        size: Math.max(span / 170, 0.05), sizeAttenuation: true,
        vertexColors: hasColor, color: hasColor ? 0xffffff : (dark ? 0xe6e6e9 : 0x17171a),
      });
      this.object = new T.Points(g, mat);
    } else {
      if (!g.hasAttribute('normal')) g.computeVertexNormals();
      const mat = new T.MeshStandardMaterial({
        color: hasColor ? 0xffffff : (dark ? 0xc9c6bf : 0xb7b4ad),
        vertexColors: hasColor, roughness: 0.62, metalness: 0.02, side: T.DoubleSide,
        wireframe: S.display === 'wire',
      });
      if (S.display === 'wire' && !hasColor) mat.color.set(dark ? 0xd8d8dc : 0x2a2a2e);
      this.object = new T.Mesh(g, mat);
    }
    this.group.add(this.object);
  },

  present(geometry, asset) {
    const T = this.THREE;
    if (this.geom && this.geom !== geometry) this.geom.dispose();
    this.geom = geometry; this.asset = asset;
    geometry.computeBoundingBox();
    if (!S.origin) {                                // one shared origin per scan: no jumping between assets
      S.origin = geometry.boundingBox.getCenter(new T.Vector3());
      this.group.position.copy(S.origin).negate();
      S.needFit = true;
    }
    this.rebuild();
    if (S.needFit || !S.userMoved) this.fit();
    S.needFit = false;

    const verts = geometry.attributes.position.count;
    const faces = geometry.index ? Math.round(geometry.index.count / 3) : 0;
    const bb = geometry.boundingBox.getSize(new T.Vector3());
    setText('vf-verts', fmtInt(verts));
    setText('vf-faces', faces ? fmtInt(faces) : '—');
    setText('vf-bbox', `${bb.x.toFixed(1)} × ${bb.y.toFixed(1)} × ${bb.z.toFixed(1)} mm`);
    $('viewer-empty').hidden = true;
    const names = { preview: 'Live point cloud', mesh: 'Reconstructed mesh', segmented: 'Segmented scan', teeth: 'Teeth only', cloud: 'Fused point cloud' };
    const label = asset.startsWith('tooth:') ? `Tooth class ${asset.slice(6).padStart(2, '0')}` : (names[asset] || asset);
    setText('vf-source', label);
    const tag = $('tag-asset'); tag.hidden = false; tag.textContent = label;
  },

  fit() {
    if (!this.object) return;
    const T = this.THREE;
    this.scene.updateMatrixWorld(true);
    const box = new T.Box3().setFromObject(this.object);
    if (box.isEmpty()) return;
    const center = box.getCenter(new T.Vector3());
    const radius = Math.max(box.getSize(new T.Vector3()).length() / 2, 0.5);
    const dist = radius / Math.sin((this.camera.fov * Math.PI / 180) / 2) * 1.08;
    this.setView(S.preset, center, dist);
  },

  setView(name, center, dist) {
    const T = this.THREE;
    if (!center) {
      const box = new T.Box3().setFromObject(this.object || this.group);
      center = box.isEmpty() ? new T.Vector3() : box.getCenter(new T.Vector3());
      dist = dist || this.camera.position.distanceTo(this.controls.target) || 60;
    }
    const dirs = {
      iso: [0.62, 0.52, 0.86], '-z': [0, 0, 1], '+z': [0, 0, -1], '-x': [-1, 0, 0], '+x': [1, 0, 0],
      '-y': [0, 1, 0.0001], '+y': [0, -1, 0.0001],           // world −Y is “up” in the scene after the half-turn
    };
    const d = new T.Vector3(...(dirs[name] || dirs.iso)).normalize();
    this.camera.position.copy(center).addScaledVector(d, dist);
    this.camera.near = Math.max(dist / 400, 0.01);
    this.camera.far = dist * 60;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
  },

  reset() {
    if (this.object) this.object.removeFromParent();
    this.clear();
    if (this.geom) { this.geom.dispose(); this.geom = null; }
    this.asset = null;
    if (this.group) this.group.position.set(0, 0, 0);
  },

  screenshot() {
    this.render();
    this.renderer.domElement.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `ioscan_view_${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    }, 'image/png');
  },
};

/* ---- model loading -------------------------------------------------------------------- */
function assetUrl(asset) {
  return asset.startsWith('tooth:') ? `/api/model/tooth/${asset.slice(6)}` : `/api/model/${asset}`;
}

function desiredAsset(snap) {
  if (S.view !== 'auto') return S.view;
  return snap.model_asset || null;
}

async function loadAsset(asset, key) {
  const token = ++S.loadToken;
  try {
    const res = await fetch(`${assetUrl(asset)}?k=${encodeURIComponent(key)}`, { cache: 'no-store' });
    if (!res.ok) { S.failKey = key; S.failAt = performance.now(); return; }   // not written yet; retry later
    const buf = await res.arrayBuffer();
    if (token !== S.loadToken) return;                                       // a newer request superseded this one
    const geom = V.loader.parse(buf);
    if (!geom.attributes.position || geom.attributes.position.count === 0) { S.failKey = key; S.failAt = performance.now(); return; }
    V.present(geom, asset);
    S.loadedKey = key;
  } catch (err) {
    console.warn('Model load failed:', err);
    S.failKey = key; S.failAt = performance.now();
  }
}

function maybeLoadModel(snap) {
  if (!V.ok) return;
  const asset = desiredAsset(snap);
  if (!asset) return;
  const key = `${asset}@${snap.model_revision}@${snap.session_dir || ''}`;
  if (key === S.loadedKey) return;
  if (key === S.failKey && performance.now() - S.failAt < 3000) return;
  loadAsset(asset, key);
}

function forceReload() { S.loadedKey = null; S.failKey = null; if (S.snap) maybeLoadModel(S.snap); }

/* ---- viewer toolbar ------------------------------------------------------------------- */
$('seg-source').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-src]'); if (!b || b.disabled) return;
  S.view = b.dataset.src;
  S.userMoved = false;
  forceReload(); renderToolbar(S.snap);
});
$('seg-display').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-disp]'); if (!b) return;
  S.display = b.dataset.disp;
  document.querySelectorAll('#seg-display button').forEach((x) => x.classList.toggle('on', x === b));
  if (V.ok) V.rebuild();
});
$('view-preset').addEventListener('change', (e) => {
  S.preset = e.target.value;
  if (V.ok) { S.userMoved = false; V.fit(); }
});
$('btn-fit').addEventListener('click', () => { S.userMoved = false; if (V.ok) V.fit(); });
$('btn-shot').addEventListener('click', () => { if (V.ok && V.object) V.screenshot(); else toast('Nothing to capture yet.'); });

/* ====================================================================== camera feed */
function setCamImage(blob) {
  const url = URL.createObjectURL(blob);
  const img = $('cam-img');
  img.onload = () => { if (S.camUrl && S.camUrl !== url) URL.revokeObjectURL(S.camUrl); S.camUrl = url; };
  img.src = url;
  if (!S.camShown) { S.camShown = true; img.hidden = false; $('cam-empty').hidden = true; }
}

async function fetchPreviewOnce() {
  try {
    const res = await fetch(`/api/camera/preview.jpg?t=${Date.now()}`, { cache: 'no-store' });
    if (res.status === 200) setCamImage(await res.blob());
  } catch (e) { /* transient; next tick retries */ }
}

async function previewLoop() {
  if (S.previewRunning) return;
  S.previewRunning = true;
  while (S.wantPreview) {
    await fetchPreviewOnce();          // strictly sequential: never queues requests behind a slow frame
    await sleep(90);
  }
  await fetchPreviewOnce();            // keep the final frame on screen
  S.previewRunning = false;
}

async function probeCamera() {
  if (S.probing) return;
  S.probing = true;
  const btns = [$('btn-probe'), $('btn-probe-empty')];
  btns.forEach((b) => { b.disabled = true; b.textContent = 'Checking…'; });
  try {
    const id = Number($('camera-id').value || 0);
    const d = await api('/api/camera/probe', { method: 'POST', body: { camera_id: id } });
    S.cam = { state: 'ready', w: d.width, h: d.height };
    await fetchPreviewOnce();
    setText('cam-res', `${d.width} × ${d.height}`);
    if (d.width !== 1600 || d.height !== 1200) {
      toast(`Camera returned ${d.width} × ${d.height}. The pipeline and calibration expect 1600 × 1200; check the UVC mode.`, 'err');
    } else {
      toast(`Camera ${d.camera_id} responds at ${d.width} × ${d.height}.`);
    }
  } catch (err) {
    S.cam = { state: 'error', w: 0, h: 0 };
    toast(err.message, 'err');
  } finally {
    S.probing = false;
    btns.forEach((b) => { b.disabled = false; b.textContent = 'Check camera'; });
    if (S.snap) render(S.snap);
  }
}
$('btn-probe').addEventListener('click', probeCamera);
$('btn-probe-empty').addEventListener('click', probeCamera);

/* ====================================================================== actions */
function showClientAlert(title, body) { S.clientAlert = { title, body }; if (S.snap) renderAlerts(S.snap); }

async function startScan() {
  if (S.starting) return;
  const payload = readForm();
  if (!payload) return;
  S.clientAlert = null;
  S.starting = true; $('btn-start').disabled = true; setText('btn-start-t', 'Starting…');
  try {
    await api('/api/capture/start', { method: 'POST', body: payload });
    // New scan: fresh view, fresh origin, fresh timers.
    V.reset?.();
    Object.assign(S, { origin: null, userMoved: false, needFit: true, loadedKey: null, failKey: null, view: 'auto',
      fps: null, fpsPrev: null, capStart: null, capAvg: null, lastGood: null, camShown: S.camShown });
    setText('vf-verts', '—'); setText('vf-faces', '—'); setText('vf-bbox', '—'); setText('vf-source', '—');
    $('tag-asset').hidden = true; $('viewer-empty').hidden = false;
    switchTab('scan');
    toast('Scan started.');
  } catch (err) {
    showClientAlert('Could not start scan', err.message);
    toast(err.message, 'err');
    switchTab('scan');
  } finally {
    S.starting = false; setText('btn-start-t', 'Start scan');
    await pollNow();
  }
}
async function stopScan() {
  try { await api('/api/capture/stop', { method: 'POST' }); toast('Stopping capture; the session will be processed.'); }
  catch (err) { toast(err.message, 'err'); }
}
async function runSegmentation() {
  S.clientAlert = null;
  try {
    await api('/api/segmentation/start', { method: 'POST', body: { jaw: $('jaw').value, models_dir: $('models-dir').value.trim() || null } });
    switchTab('scan');
  } catch (err) { showClientAlert('Could not start segmentation', err.message); toast(err.message, 'err'); }
  await pollNow();
}
$('btn-start').addEventListener('click', startScan);
$('btn-stop').addEventListener('click', stopScan);
$('btn-segment').addEventListener('click', runSegmentation);
$('btn-show-all').addEventListener('click', () => { S.view = 'segmented'; forceReload(); renderToolbar(S.snap); renderResults(S.snap); });

document.addEventListener('click', async (e) => {
  const c = e.target.closest('[data-copy]');
  if (c) {
    const text = $(c.dataset.copy).title;
    try { await navigator.clipboard.writeText(text); toast('Path copied.'); } catch (_) { toast('Copy is unavailable in this browser context.', 'err'); }
  }
  const v = e.target.closest('[data-view]');
  if (v) { S.view = v.dataset.view; S.userMoved = false; forceReload(); renderToolbar(S.snap); renderResults(S.snap); }
});

window.addEventListener('beforeunload', (e) => {
  const p = S.snap && S.snap.phase;
  if (p === 'capturing' || p === 'processing') { e.preventDefault(); e.returnValue = ''; }
});

/* ====================================================================== status → UI */
const STAGE_TITLES = {
  ready: 'System ready', capturing: 'Recording scan', captured: 'Capture saved', reconstructing: 'Triangulating and fusing profiles',
  meshing: 'Building surface mesh', mesh_ready: 'Surface mesh ready', reconstructed: 'Reconstruction complete',
  segmenting: 'Segmenting teeth', complete: 'Processing complete', segmentation_error: 'Segmentation needs review', error: 'Stopped with an error',
};

function stepStates(snap) {
  const { phase, stage } = snap;
  const out = snap.outputs || {};
  const hasMesh = !!out.mesh;
  const recStages = ['reconstructing', 'meshing', 'mesh_ready'];
  const failedAt = S.lastGood;
  const errored = phase === 'error';

  let cap;
  if (phase === 'capturing') cap = ['active', 'Recording'];
  else if (errored && (failedAt === 'capturing' || !failedAt)) cap = ['error', 'Failed'];
  else if (phase === 'idle') cap = ['', 'Idle'];
  else cap = ['done', `${fmtInt(snap.frame)} frames`];

  let rec;
  if (recStages.includes(stage)) rec = ['active', 'Running'];
  else if (['reconstructed', 'segmenting', 'complete', 'segmentation_error'].includes(stage) || hasMesh) rec = ['done', 'Done'];
  else if (errored && (failedAt === 'captured' || recStages.includes(failedAt))) rec = ['error', 'Failed'];
  else rec = ['', 'Waiting'];

  let seg;
  if (stage === 'segmenting') seg = ['active', 'Running'];
  else if (snap.segmentation) seg = ['done', 'Done'];
  else if (stage === 'segmentation_error' || (phase === 'complete_with_warnings' && hasMesh)) seg = ['error', 'Needs review'];
  else if (snap.auto_segment === false && (phase === 'complete')) seg = ['skipped', 'Skipped'];
  else if (snap.auto_segment === false && phase !== 'idle') seg = ['skipped', 'Off'];
  else seg = ['', 'Waiting'];
  return { cap, rec, seg };
}

function renderWorkflow(snap) {
  const s = stepStates(snap);
  for (const [id, st] of [['capture', s.cap], ['reconstruct', s.rec], ['segment', s.seg]]) {
    const el = $(`wf-${id}`);
    el.className = `wf-step ${st[0]}`;
    setText(`wf-${id}-s`, st[1]);
  }
  const rows = [['st-capture', s.cap, { '': 'READY', active: 'LIVE', done: 'DONE', error: 'FAILED' }],
                ['st-reconstruct', s.rec, { '': 'WAITING', active: 'RUNNING', done: 'DONE', error: 'FAILED' }],
                ['st-segment', s.seg, { '': 'WAITING', active: 'RUNNING', done: 'DONE', error: 'REVIEW', skipped: 'SKIPPED' }]];
  for (const [id, st, labels] of rows) {
    const li = $(id);
    li.className = st[0];
    li.querySelector('.st').textContent = labels[st[0]] || '';
  }
}

function renderChips(snap) {
  const phase = snap.phase, streaming = phase === 'capturing' && snap.frame > 0;
  const err = (snap.error || '');
  const failedCap = phase === 'error' && (S.lastGood === 'capturing' || !S.lastGood);

  // Service
  $('dot-server').className = `dot ${S.connected ? 'ok' : 'err'}`;
  $('chip-server').lastChild.textContent = S.connected ? 'Service' : 'Service offline';

  // Camera
  let cls = '', txt = 'Camera · not tested';
  if (streaming) { cls = 'live'; txt = 'Camera · streaming'; }
  else if (phase === 'capturing') { cls = 'warn'; txt = 'Camera · opening'; }
  else if (failedCap && /camera|frame/i.test(err)) { cls = 'err'; txt = 'Camera · fault'; }
  else if (S.cam.state === 'ready') { cls = 'ok'; txt = `Camera · ${S.cam.w}×${S.cam.h}`; }
  else if (S.cam.state === 'error') { cls = 'err'; txt = 'Camera · no signal'; }
  $('dot-camera').className = `dot ${cls}`; setText('txt-camera', txt);

  // IMU / ESP32 (inferred from the capture stream; the service has no separate serial probe)
  cls = ''; txt = 'IMU · idle';
  if (streaming) { cls = 'ok'; txt = 'IMU · streaming'; }
  else if (phase === 'capturing') { cls = 'warn'; txt = 'IMU · waiting'; }
  else if (failedCap && /esp32|serial|bno|strobe|imu|COM\d/i.test(err)) { cls = 'err'; txt = 'IMU · fault'; }
  $('dot-imu').className = `dot ${cls}`; setText('txt-imu', txt);
}

function updateFps(snap) {
  const now = performance.now();
  if (snap.phase === 'capturing') {
    if (S.capStart === null) S.capStart = now - (snap.frame / TARGET_FPS) * 1000;
    if (S.fpsPrev && snap.frame > S.fpsPrev.frame) {
      const inst = (snap.frame - S.fpsPrev.frame) / ((now - S.fpsPrev.t) / 1000);
      S.fps = S.fps == null ? inst : S.fps * 0.7 + inst * 0.3;
    }
    S.fpsPrev = { t: now, frame: snap.frame };
  } else if (S.fpsPrev && S.capStart !== null && S.capAvg === null && snap.frame > 0) {
    const secs = (S.fpsPrev.t - S.capStart) / 1000;
    S.capAvg = secs > 1 ? snap.frame / secs : null;
    S.fpsPrev = null;
  }
}

function renderReadouts(snap) {
  const capturing = snap.phase === 'capturing';
  setText('ro-frames', snap.frame ? `${fmtInt(snap.frame)} / ${fmtInt(snap.frames_total)}` : '—');
  setText('ro-frames-sub', capturing ? `Recording · ${S.capStart !== null ? fmtClock(performance.now() - S.capStart) : '--:--'}` : snap.frame ? 'Capture saved' : 'Not recording');

  const fps = capturing ? S.fps : S.capAvg;
  const low = capturing && fps != null && snap.frame > 60 && fps < TARGET_FPS * 0.9;
  $('ro-fps').firstChild.nodeValue = fps != null ? fps.toFixed(1) : '—';
  const sub = $('ro-fps-sub');
  sub.className = `sub ${low ? 'warn' : ''}`;
  sub.textContent = low ? `Below ${TARGET_FPS} fps target`
    : capturing ? (fps == null ? 'Measuring…' : `Target ${TARGET_FPS} fps`)
    : S.capAvg != null ? 'Session average' : `Target ${TARGET_FPS} fps`;

  const live = snap.model_asset === 'preview' && snap.phase !== 'complete';
  setText('ro-points-label', live ? 'Live points' : 'Fused points');
  setText('ro-points', snap.point_count ? fmtInt(snap.point_count) : '—');
  setText('ro-points-sub', live ? 'Raw triangulated' : 'After voxel fusion');
  setText('ro-faces', snap.mesh_faces ? fmtInt(snap.mesh_faces) : '—');
  setText('ro-faces-sub', snap.mesh_vertices ? `${fmtInt(snap.mesh_vertices)} vertices` : 'No surface yet');
  $('ro-path').firstChild.nodeValue = snap.trajectory_mm ? Number(snap.trajectory_mm).toFixed(1) : '—';
  setText('ro-path-sub', snap.trajectory_mm ? 'Estimated at assumed speed' : 'Estimated trajectory');
}

function renderCameraMeta(snap) {
  const capturing = snap.phase === 'capturing';
  $('cam-hud').hidden = !(S.camShown || snap.frame > 0);
  $('hud-rec').firstElementChild.className = `dot ${capturing ? 'rec' : ''}`;
  setText('hud-rec-t', capturing ? 'REC' : snap.frame ? 'Captured' : 'Standby');
  setText('hud-laser', `Last frame · laser ${snap.frame ? (snap.laser_on ? 'ON' : 'OFF') : '—'}`);
  setText('cam-meta-r', `Frame ${snap.frame ? fmtInt(snap.frame) : '—'} / ${snap.frames_total ? fmtInt(snap.frames_total) : '—'}`);
  setText('cam-meta-l', capturing ? 'Streaming from scanner'
    : snap.frame ? 'Last captured frame'
    : S.cam.state === 'ready' ? 'Camera test passed'
    : S.cam.state === 'error' ? 'Camera test failed' : 'Camera not tested');
}

function renderProgress(snap) {
  const stage = snap.stage || 'ready';
  const p = Math.max(0, Math.min(100, Number(snap.progress || 0)));
  const idle = snap.phase === 'idle';
  setText('prog-title', STAGE_TITLES[stage] || 'System ready');
  setText('prog-pct', idle ? '0%' : `${Math.round(p)}%`);
  $('progress-bar').style.width = idle ? '0%' : `${p}%`;
  const total = snap.frames_total || 0;
  let count = '—';
  if (stage === 'capturing' && total) count = `${fmtInt(snap.frame)} / ${fmtInt(total)} frames`;
  else if ((stage === 'reconstructing' || stage === 'meshing') && total) count = `${fmtInt(snap.processed_frames || 0)} / ${fmtInt(total)} frames`;
  setText('prog-count', count);
  const msg = snap.error ? `${snap.message} ${snap.error}` : (snap.message || '');
  setText('status-msg', msg);
  setText('foot-msg', snap.error ? snap.error : (snap.message || 'Ready'));
}

function hintFor(error, where) {
  const e = String(error || '');
  if (/queue overflow/i.test(e)) return 'The host did not keep up with the camera strobe rate. Close other heavy applications and retry; if it persists the capture loop needs more headroom.';
  if (/sequence gap/i.test(e)) return 'Serial packets were lost between the ESP32 and the PC. Check the USB cable/port and that nothing else is reading the port.';
  if (/No synchronized ESP32|strobe packet/i.test(e)) return 'No camera-strobe packets arrived. Check the STROBE wire to GPIO34, the BNO086 connection, and that the firmware is running.';
  if (/could not open usb camera|camera stopped/i.test(e)) return 'The camera could not be opened or stopped delivering frames. Close other apps using it and check the camera index.';
  if (/No laser points/i.test(e)) return 'No laser line was found. Check the calibration file, laser alignment, and that saved frames alternate laser OFF / ON in step with the firmware (a one-frame strobe offset produces exactly this error).';
  if (/PyTorch|checkpoint|MeshSegNet/i.test(e)) return 'Check that PyTorch is installed and that the model folder contains the checkpoint for the selected jaw.';
  if (/Calibration JSON/i.test(e)) return 'Check the calibration file path and that it contains both “camera” and “laser” objects.';
  return null;
}

function renderAlerts(snap) {
  const items = [];
  if (S.clientAlert) items.push({ kind: 'err', title: S.clientAlert.title, body: S.clientAlert.body, hint: hintFor(S.clientAlert.body), dismiss: true });
  if (snap.error) {
    const where = snap.stage === 'segmentation_error' || snap.phase === 'complete_with_warnings' ? 'Segmentation'
      : (S.lastGood === 'capturing' || !S.lastGood) ? 'Capture' : 'Reconstruction';
    items.push({ kind: snap.phase === 'complete_with_warnings' ? 'warn' : 'err', title: `${where} problem`, body: snap.error, hint: hintFor(snap.error, where) });
  }
  if (snap.phase === 'capturing' && S.fps != null && snap.frame > 60 && S.fps < TARGET_FPS * 0.9) {
    items.push({ kind: 'warn', title: 'Capture rate below target', body: `Measured ${S.fps.toFixed(1)} fps against a ${TARGET_FPS} fps strobe. If the host falls behind, camera frames and strobe packets can fall out of step.` });
  }
  $('sec-alerts').hidden = items.length === 0;
  $('alerts').innerHTML = items.map((a, i) => `
    <div class="alert ${a.kind}">
      <svg class="i"><use href="#${a.kind === 'err' ? 'i-alert' : 'i-info'}"/></svg>
      <div class="body"><b>${esc(a.title)}</b>${esc(a.body)}${a.hint ? `<div style="margin-top:4px;color:var(--text-2)">${esc(a.hint)}</div>` : ''}</div>
      ${a.dismiss ? `<button class="x" data-dismiss="${i}" aria-label="Dismiss"><svg class="i"><use href="#i-x"/></svg></button>` : '<span></span>'}
    </div>`).join('');
}
$('alerts').addEventListener('click', (e) => { if (e.target.closest('[data-dismiss]')) { S.clientAlert = null; if (S.snap) renderAlerts(S.snap); } });

function classifyWarning(w) {
  const l = w.toLowerCase();
  if (l.includes('nominal')) return ['Calibration', true];
  if (l.includes('translation is assumed') || l.includes('assumed')) return ['Pose', true];
  if (l.includes('trajectory is only') || l.includes('local patch')) return ['Coverage', true];
  if (l.includes('single laser pass')) return ['Coverage', false];
  if (l.includes('simulated')) return ['Source', false];
  if (l.includes('live 3d preview') || l.includes('live preview')) return ['Preview', false];
  if (l.includes('partial') || l.includes('small')) return ['Coverage', true];
  return ['Notice', false];
}

function renderWarnings(snap) {
  const list = [...new Set((snap.warnings || []).filter(Boolean))];
  $('sec-warnings').hidden = list.length === 0;
  setText('warn-count', list.length ? `${list.length}` : '');
  $('warn-list').innerHTML = list.map((w) => {
    const [k, strong] = classifyWarning(w);
    return `<li><span class="k ${strong ? '' : 'info'}">${k}</span><span>${esc(w)}</span></li>`;
  }).join('');
  $('tag-nominal').hidden = !list.some((w) => /nominal/i.test(w));
}

function renderToolbar(snap) {
  if (!snap) return;
  const out = snap.outputs || {};
  const avail = {
    auto: true,
    preview: !!snap.model_asset || snap.point_count > 0,
    mesh: !!out.mesh || ['mesh', 'segmented'].includes(snap.model_asset),
    segmented: !!out.segmented,
    teeth: !!out.teeth,
  };
  document.querySelectorAll('#seg-source button').forEach((b) => {
    const k = b.dataset.src;
    b.disabled = !avail[k];
    b.classList.toggle('on', S.view === k);
  });
}

function renderViewerEmpty(snap) {
  if (!V.ok || V.object) return;
  const capturing = snap.phase === 'capturing';
  setText('viewer-empty-title', capturing ? 'Waiting for the first laser profiles' : 'No geometry');
  setText('viewer-empty-sub', capturing
    ? 'The fused point cloud appears after laser-ON frames pair with an ambient frame and an IMU orientation.'
    : 'The fused point cloud appears here while the scan is recording.');
}

function renderResults(snap) {
  const out = snap.outputs || {};
  const keys = ['mesh', 'segmented', 'teeth', 'cloud'].filter((k) => out[k]);
  const seg = snap.segmentation;
  const teeth = seg ? Object.keys(seg.per_tooth_meshes || {}).sort((a, b) => Number(a) - Number(b)) : [];
  const total = keys.length + teeth.length;
  setText('file-count', `${total} file${total === 1 ? '' : 's'}`);
  const badge = $('export-count'); badge.hidden = total === 0; badge.textContent = total;

  $('file-list').innerHTML = keys.length ? `<div class="file-list">${keys.map((k) => {
    const viewable = ['mesh', 'segmented', 'teeth'].includes(k);
    return `<div class="file">
      <div class="name">${FILES[k].label}<small class="mono">${FILES[k].file}</small></div>
      ${viewable ? `<button class="link-btn" data-view="${k}">View</button>` : '<span></span>'}
      <a class="link-btn" href="/api/model/${k}" download="${FILES[k].file}" title="Download PLY"><svg class="i"><use href="#i-down"/></svg>PLY</a>
    </div>`; }).join('')}</div>`
    : '<div class="empty">Reconstruction outputs appear here once processing finishes.</div>';

  $('sec-seg').hidden = !seg;
  if (seg) {
    const rows = [
      ['Jaw', seg.jaw], ['Teeth coverage', `${seg.teeth_pct} %`], ['Gingiva', `${seg.gingiva_pct} %`],
      ['Classes found', `${teeth.length} of 14`], ['Faces (input)', fmtInt(seg.original_cells)],
      ['Device', seg.device], ['Checkpoint', basename(seg.checkpoint)],
    ];
    $('seg-summary').innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td>${esc(b)}</td></tr>`).join('');
    const counts = seg.class_face_counts || {};
    $('tooth-list').innerHTML = teeth.length ? teeth.map((id) => {
      const c = SEG_PALETTE[Number(id)] || [128, 128, 128];
      const sel = S.view === `tooth:${id}`;
      return `<div class="tooth ${sel ? 'sel' : ''}">
        <span class="swatch" style="background:rgb(${c[0]},${c[1]},${c[2]})"></span>
        <span>Tooth class ${String(id).padStart(2, '0')} <span class="mono" style="color:var(--muted)">· ${fmtInt(counts[id])} faces</span></span>
        <button class="link-btn" data-view="tooth:${id}">${sel ? 'Showing' : 'Isolate'}</button>
        <a class="link-btn" href="/api/model/tooth/${id}" download="tooth_class_${String(id).padStart(2, '0')}.ply" title="Download PLY" aria-label="Download tooth class ${id}"><svg class="i"><use href="#i-down"/></svg></a>
      </div>`; }).join('') : '<div class="empty" style="border:0">No tooth classes were predicted for this scan.</div>';
    $('btn-show-all').hidden = !S.view.startsWith('tooth:');
  }

  $('sec-paths').hidden = !snap.session_dir;
  if (snap.session_dir) {
    for (const [id, val] of [['path-session', snap.session_dir], ['path-results', snap.results_dir]]) {
      const el = $(id); el.textContent = '\u200E' + (val || ''); el.title = val || '';
    }
  }
}

function renderControls(snap) {
  const busy = ['capturing', 'processing'].includes(snap.phase);
  $('btn-start').disabled = busy || S.starting;
  $('btn-stop').disabled = snap.phase !== 'capturing';
  $('btn-segment').disabled = !(snap.outputs && snap.outputs.mesh) || busy;
  lockForm(busy);
  document.title = snap.phase === 'capturing' ? `● REC ${Math.round(snap.progress || 0)}% · IOSCAN`
    : snap.phase === 'processing' ? `${STAGE_TITLES[snap.stage] || 'Processing'} · IOSCAN` : 'IOSCAN · Scan Console';
}

function render(snap) {
  S.snap = snap;
  if (snap.stage && snap.stage !== 'error' && snap.stage !== 'ready') S.lastGood = snap.stage;
  if (snap.phase === 'idle') S.lastGood = null;

  // A different session means a new scan was started elsewhere (or the service restarted).
  const sk = snap.session_dir || null;
  if (sk !== S.sessionKey) {
    if (S.sessionKey !== null || sk !== null) { S.origin = null; S.loadedKey = null; S.needFit = true; S.userMoved = false; if (V.ok) V.reset(); }
    S.sessionKey = sk;
  }

  updateFps(snap);
  S.wantPreview = snap.phase === 'capturing';
  if (S.wantPreview) previewLoop();

  renderChips(snap);
  renderWorkflow(snap);
  renderReadouts(snap);
  renderCameraMeta(snap);
  renderProgress(snap);
  renderAlerts(snap);
  renderWarnings(snap);
  renderToolbar(snap);
  renderViewerEmpty(snap);
  renderResults(snap);
  renderControls(snap);
  maybeLoadModel(snap);
}

/* ====================================================================== polling */
let pollTimer = null;
async function pollNow() { clearTimeout(pollTimer); await poll(); }
async function poll() {
  try {
    const snap = await api('/api/status');
    if (!S.connected) toast('Reconnected to the scanner service.');
    S.connected = true; S.failures = 0;
    $('offline-banner').hidden = true;
    render(snap);
  } catch (err) {
    S.failures += 1;
    if (S.failures >= 3) {
      S.connected = false;
      $('offline-banner').hidden = false;
      $('dot-server').className = 'dot err';
      $('chip-server').lastChild.textContent = 'Service offline';
    }
  } finally {
    const capturing = S.snap && S.snap.phase === 'capturing';
    pollTimer = setTimeout(poll, !S.connected ? 2000 : capturing ? 400 : 800);
  }
}

/* ====================================================================== boot */
(async function boot() {
  loadSetup();
  applyTheme(document.documentElement.getAttribute('data-theme') || 'light');
  poll();
  await V.init();
  if (V.ok) { V.applyTheme(); if (S.snap) maybeLoadModel(S.snap); }
})();