/* IOSCAN Scan Console — front-end controller.
 *
 * Talks to the existing FastAPI service only through its documented endpoints:
 *   GET  /api/status                 POST /api/capture/start    POST /api/capture/stop
 *   POST /api/camera/probe           POST /api/segmentation/start
 *   GET  /api/sessions               POST /api/process/start    POST /api/stitch/start
 *   GET  /api/calibration            POST /api/calibration/start  POST /api/results/open
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
  if (name === 'library') refreshSessions();
}
/* Setup is an accordion: opening one block closes the others (also where <details name> is unsupported). */
const ACC = [...document.querySelectorAll('#pane-setup details.acc')];
ACC.forEach((d) => d.addEventListener('toggle', () => { if (d.open) ACC.forEach((o) => { if (o !== d) o.open = false; }); }));
function openSetup(id) { switchTab('setup'); const d = $(id); if (d) { d.open = true; d.scrollIntoView({ block: 'nearest' }); } }
$('btn-guide').addEventListener('click', () => $('guide-dialog').showModal());
$('btn-guide-close').addEventListener('click', () => $('guide-dialog').close());
$('guide-dialog').addEventListener('click', (e) => { if (e.target === $('guide-dialog')) $('guide-dialog').close(); });
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => switchTab(t.dataset.tab)));

/* ====================================================================== setup form */
const FORM_IDS = ['camera-device', 'exposure', 'serial-port', 'session-name', 'frame-count', 'speed', 'baseline-mm', 'laser-tilt-deg', 'fan-angle-deg', 'depth-min', 'depth-max', 'jaw', 'mesh-method', 'calibration-file', 'models-dir', 'auto-segment',
  'recon-mode', 'translation', 'scan-length', 'secondary-ratio', 'min-prominence', 'auto-trim', 'auto-depth'];
const PERSIST = FORM_IDS.filter((i) => !['session-name', 'camera-device'].includes(i));

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
function updateSetupSummaries() {
  let cam = null;
  try { cam = JSON.parse($('camera-device').value || 'null'); } catch (e) { cam = null; }
  setText('sum-devices', cam ? `${cam.name} · ${$('serial-port').value.trim() || '—'}` : 'No camera selected');
  const frames = Number($('frame-count').value) || 0;
  setText('sum-scan', `${$('jaw').value === 'upper' ? 'Upper' : 'Lower'} jaw · up to ${Math.round(frames / 50)} s`);
}
FORM_IDS.forEach((id) => $(id).addEventListener('change', () => { saveSetup(); updateSetupSummaries(); }));

function updateWorkingDistance() {
  const baseline = Number($('baseline-mm').value);
  const tilt = Number($('laser-tilt-deg').value);
  const distance = baseline > 0 && tilt > 0 && tilt < 90
    ? baseline / Math.tan(tilt * Math.PI / 180) : NaN;
  $('working-distance').value = Number.isFinite(distance) ? distance.toFixed(2) : '—';
}
$('baseline-mm').addEventListener('input', updateWorkingDistance);
$('laser-tilt-deg').addEventListener('input', updateWorkingDistance);

function hardwareGeometryPayload() {
  return {
    baseline_mm: Number($('baseline-mm').value),
    laser_tilt_deg: Number($('laser-tilt-deg').value),
    fan_half_angle_deg: Number($('fan-angle-deg').value),
    exposure: Number($('exposure').value),
    speed_mm_s: Number($('speed').value),
    depth_min_mm: Number($('depth-min').value),
    depth_max_mm: Number($('depth-max').value),
    ...reconSettings(),
  };
}

/* Reconstruction settings shared by capture, reprocessing and the saved hardware settings. */
function reconSettings() {
  return {
    recon_mode: $('recon-mode').value,
    translation: $('translation').value,
    scan_length_mm: Number($('scan-length').value) || 0,
    auto_trim: $('auto-trim').checked,
    auto_depth: $('auto-depth').checked,
    secondary_ratio: Number($('secondary-ratio').value),
    min_prominence: Number($('min-prominence').value),
  };
}
function validRecon(r) {
  let ok = true;
  if (!Number.isFinite(r.scan_length_mm) || r.scan_length_mm < 0 || r.scan_length_mm > 500) { fieldError('scan-length', 'Scan length: 0 (unknown) to 500 mm'); ok = false; } else fieldError('scan-length', null);
  if (!Number.isFinite(r.secondary_ratio) || r.secondary_ratio < 0.1 || r.secondary_ratio > 1) { fieldError('secondary-ratio', 'Ratio: 0.1 to 1.0'); ok = false; } else fieldError('secondary-ratio', null);
  if (!Number.isFinite(r.min_prominence) || r.min_prominence < 3 || r.min_prominence > 250) { fieldError('min-prominence', 'Brightness: 3 to 250 DN'); ok = false; } else fieldError('min-prominence', null);
  if (r.translation === 'length' && !(r.scan_length_mm > 0)) { fieldError('scan-length', 'Enter the pass length for “Known scan length”'); ok = false; }
  return ok;
}
function updateReconFields() {
  const sweep = $('recon-mode').value === 'sweep';
  $('translation').disabled = !sweep;
  $('auto-trim').disabled = !sweep;
  $('mesh-method').disabled = sweep;
  const manualDepth = !$('auto-depth').checked;
  $('depth-min').disabled = !manualDepth; $('depth-max').disabled = !manualDepth;
}
['recon-mode', 'auto-depth'].forEach((id) => $(id).addEventListener('change', updateReconFields));

async function loadHardwareGeometry() {
  try {
    const cfg = await api('/api/hardware-geometry');
    $('baseline-mm').value = cfg.baseline_mm;
    $('laser-tilt-deg').value = cfg.laser_tilt_deg;
    $('fan-angle-deg').value = cfg.fan_half_angle_deg;
    $('exposure').value = cfg.exposure;
    $('speed').value = cfg.speed_mm_s;
    $('depth-min').value = cfg.depth_min_mm;
    $('depth-max').value = cfg.depth_max_mm;
    if (cfg.recon_mode) $('recon-mode').value = cfg.recon_mode;
    if (cfg.translation) $('translation').value = cfg.translation;
    if (cfg.scan_length_mm !== undefined) $('scan-length').value = cfg.scan_length_mm;
    if (cfg.secondary_ratio !== undefined) $('secondary-ratio').value = cfg.secondary_ratio;
    if (cfg.min_prominence !== undefined) $('min-prominence').value = cfg.min_prominence;
    if (cfg.auto_trim !== undefined) $('auto-trim').checked = !!cfg.auto_trim;
    if (cfg.auto_depth !== undefined) $('auto-depth').checked = !!cfg.auto_depth;
    updateReconFields();
    updateWorkingDistance();
    saveSetup();
    setText('geometry-status', `Saved hardware settings loaded · calculated working distance ${cfg.working_distance_mm.toFixed(2)} mm.`);
  } catch (err) {
    updateWorkingDistance();
    setText('geometry-status', `Could not load saved hardware settings: ${err.message}`);
  }
}

$('btn-save-geometry').addEventListener('click', async () => {
  const btn = $('btn-save-geometry');
  const data = hardwareGeometryPayload();
  const valid = Number.isFinite(data.baseline_mm) && data.baseline_mm > 0.1 && data.baseline_mm <= 100
    && Number.isFinite(data.laser_tilt_deg) && data.laser_tilt_deg > 1 && data.laser_tilt_deg < 89
    && Number.isFinite(data.fan_half_angle_deg) && data.fan_half_angle_deg > 1 && data.fan_half_angle_deg < 89
    && Number.isInteger(data.exposure) && data.exposure >= -13 && data.exposure <= 0
    && Number.isFinite(data.speed_mm_s) && data.speed_mm_s > 0.05 && data.speed_mm_s <= 50
    && Number.isFinite(data.depth_min_mm) && Number.isFinite(data.depth_max_mm)
    && data.depth_min_mm >= 0.1 && data.depth_max_mm > data.depth_min_mm && data.depth_max_mm <= 1000;
  if (!valid || !validRecon(data)) { toast('Check the hardware geometry and reconstruction values.', 'err'); return; }
  btn.disabled = true;
  try {
    const saved = await api('/api/hardware-geometry', { method: 'POST', body: data });
    updateWorkingDistance();
    saveSetup();
    setText('geometry-status', `Saved to configs/hardware_geometry.json · calculated working distance ${saved.working_distance_mm.toFixed(2)} mm.`);
    toast('Hardware geometry and scan settings saved.');
  } catch (err) {
    setText('geometry-status', `Could not save settings: ${err.message}`);
    toast(err.message, 'err');
  } finally { btn.disabled = false; }
});

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
  let camera = null;
  try { camera = JSON.parse($('camera-device').value || 'null'); } catch (e) { camera = null; }
  if (!camera || !Number.isInteger(camera.camera_id) || !camera.camera_backend || !camera.name) {
    toast('Refresh the camera list and select the physical Arducam UVC device by name.', 'err');
    openSetup('acc-devices');
    return null;
  }
  const exposure = intIn('exposure', -13, 0, 'Exposure');
  const frames = intIn('frame-count', 20, 20000, 'Frame limit');

  const speed = Number($('speed').value);
  if (!Number.isFinite(speed) || speed <= 0.05 || speed > 50) { fieldError('speed', 'Speed: above 0.05 up to 50 mm/s'); ok = false; } else fieldError('speed', null);

  const baseline = Number($('baseline-mm').value);
  if (!Number.isFinite(baseline) || baseline <= 0.1 || baseline > 100) { fieldError('baseline-mm', 'Baseline must be above 0.1 and at most 100 mm'); ok = false; } else fieldError('baseline-mm', null);
  const tilt = Number($('laser-tilt-deg').value);
  if (!Number.isFinite(tilt) || tilt <= 1 || tilt >= 89) { fieldError('laser-tilt-deg', 'Angle must be between 1° and 89°'); ok = false; } else fieldError('laser-tilt-deg', null);
  const fan = Number($('fan-angle-deg').value);
  if (!Number.isFinite(fan) || fan <= 1 || fan >= 89) { fieldError('fan-angle-deg', 'Fan half-angle must be between 1° and 89°'); ok = false; } else fieldError('fan-angle-deg', null);
  const depthMin = Number($('depth-min').value), depthMax = Number($('depth-max').value);
  if (!Number.isFinite(depthMin) || depthMin < 0.1 || depthMin >= depthMax) { fieldError('depth-min', 'Minimum depth must be positive and below maximum depth'); ok = false; } else fieldError('depth-min', null);
  if (!Number.isFinite(depthMax) || depthMax <= depthMin || depthMax > 1000) { fieldError('depth-max', 'Maximum depth must exceed minimum depth and be at most 1000 mm'); ok = false; } else fieldError('depth-max', null);

  const serial_port = $('serial-port').value.trim();
  if (!serial_port || serial_port.length > 32) { fieldError('serial-port', 'Enter the ESP32 serial port, e.g. COM3'); ok = false; } else fieldError('serial-port', null);

  const session_name = $('session-name').value.trim();
  if (session_name && !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(session_name)) {
    fieldError('session-name', 'Letters, numbers, “_” and “-” only (max 64)'); ok = false;
  } else fieldError('session-name', null);

  const recon = reconSettings();
  if (!validRecon(recon)) ok = false;
  if (!ok) { switchTab('setup'); return null; }
  const payload = {
    ...recon,
    camera_id: camera.camera_id, camera_backend: camera.camera_backend, camera_name: camera.name,
    serial_port, frames, exposure, speed_mm_s: speed,
    baseline_mm: baseline, laser_tilt_deg: tilt, fan_half_angle_deg: fan,
    depth_min_mm: depthMin, depth_max_mm: depthMax,
    jaw: $('jaw').value, mesh_method: $('mesh-method').value,
    auto_segment: $('auto-segment').checked,
    models_dir: $('models-dir').value.trim() || null,
    calibration_file: $('calibration-file').value.trim() || null,
  };
  if (session_name) payload.session_name = session_name;
  return payload;
}

let formLocked = null;
function lockForm(locked) {
  if (formLocked === locked) return;   // render() runs every poll; do not undo field-level disabling
  formLocked = locked;
  document.querySelectorAll('#pane-setup input, #pane-setup select, #pane-setup button').forEach((el) => { el.disabled = locked; });
  if (!locked) updateReconFields();
  renderLibrary();
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
  const roiImg = $('roi-preview-img');
  if (roiImg) roiImg.src = url;
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
  // Keep polling whenever the server has a live camera (also after a scan, while it is processed).
  while (S.wantPreview || (S.snap ? S.snap.camera_live : S.cam.state === 'ready')) {
    await fetchPreviewOnce();          // strictly sequential: never queues requests behind a slow frame
    await sleep(90);
  }
  await fetchPreviewOnce();            // keep the final frame on screen
  S.previewRunning = false;
}

/* ====================================================================== operator-drawn camera ROI */
const ROI = { points: [], enabled: false, dragIndex: null, loaded: false, dirty: false };

function roiCoords() {
  return ROI.points.map(([x, y]) => `${(x * 1600).toFixed(1)},${(y * 1200).toFixed(1)}`).join(' ');
}

function renderROI() {
  const points = ROI.points;
  const polygon = $('roi-shape');
  const outside = $('roi-outside');
  const handles = $('roi-handles');
  const valid = points.length >= 3;
  polygon.setAttribute('points', roiCoords());
  polygon.style.display = points.length >= 2 ? '' : 'none';
  if (ROI.enabled && valid) {
    const inner = points.map(([x, y]) => `${(x * 1600).toFixed(1)} ${(y * 1200).toFixed(1)}`).join(' L');
    outside.setAttribute('d', `M0 0H1600V1200H0Z M${inner}Z`);
    outside.style.display = '';
  } else {
    outside.setAttribute('d', '');
    outside.style.display = 'none';
  }
  handles.replaceChildren();
  points.forEach(([x, y], index) => {
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('class', 'roi-vertex'); c.setAttribute('data-index', String(index));
    c.setAttribute('cx', String(x * 1600)); c.setAttribute('cy', String(y * 1200)); c.setAttribute('r', '20');
    c.setAttribute('aria-label', `ROI vertex ${index + 1}`);
    handles.appendChild(c);
  });
  setText('roi-vertex-count', `${points.length} ${points.length === 1 ? 'vertex' : 'vertices'}`);
  $('roi-enabled').checked = ROI.enabled;
  $('btn-roi-undo').disabled = points.length === 0;
  $('btn-roi-save').disabled = ROI.enabled && points.length < 3;
  $('roi-empty-note').textContent = points.length
    ? 'Click to add another corner · drag a handle to adjust · Save ROI when finished.'
    : 'Check the camera, then click around the inside edge of the opening.';
  if (ROI.dirty) setText('roi-status', 'Unsaved ROI edits · save to apply this polygon to future scans.');
  else if (!ROI.loaded) setText('roi-status', 'Draw an ROI, then save it.');
}

async function loadROI() {
  try {
    const data = await api('/api/roi');
    ROI.points = Array.isArray(data.points) ? data.points.map((p) => [Number(p[0]), Number(p[1])]) : [];
    ROI.enabled = !!data.enabled;
    ROI.loaded = true; ROI.dirty = false;
    renderROI();
    setText('roi-status', ROI.enabled
      ? `Saved and active · ${ROI.points.length} vertices · applied before laser-line extraction.`
      : ROI.points.length ? `Saved but inactive · ${ROI.points.length} vertices · full frame will be processed.`
        : 'No ROI saved · full camera frame will be processed.');
  } catch (err) {
    setText('roi-status', `Could not load saved ROI: ${err.message}`);
    toast(`Could not load camera ROI: ${err.message}`, 'err');
  }
}

function roiPointerPosition(event) {
  const rect = $('roi-overlay').getBoundingClientRect();
  return [
    Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  ];
}

$('roi-overlay').addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  const handle = event.target.closest && event.target.closest('circle[data-index]');
  if (handle) {
    ROI.dragIndex = Number(handle.dataset.index);
    $('roi-overlay').setPointerCapture(event.pointerId);
    event.preventDefault();
    return;
  }
  ROI.points.push(roiPointerPosition(event));
  ROI.loaded = true; ROI.dirty = true;
  renderROI();
});
$('roi-overlay').addEventListener('pointermove', (event) => {
  if (ROI.dragIndex === null) return;
  ROI.points[ROI.dragIndex] = roiPointerPosition(event);
  ROI.dirty = true; renderROI();
});
function finishROIDrag() { ROI.dragIndex = null; }
$('roi-overlay').addEventListener('pointerup', finishROIDrag);
$('roi-overlay').addEventListener('pointercancel', finishROIDrag);
$('btn-roi-undo').addEventListener('click', () => { ROI.points.pop(); ROI.dirty = true; renderROI(); });
$('btn-roi-clear').addEventListener('click', () => {
  ROI.points = []; ROI.enabled = false; ROI.loaded = true; ROI.dirty = true; renderROI();
  setText('roi-status', 'ROI cleared in the editor. Save ROI to disable the mask for future scans.');
});
$('roi-enabled').addEventListener('change', (event) => { ROI.enabled = !!event.target.checked; ROI.loaded = true; ROI.dirty = true; renderROI(); });
$('btn-roi-save').addEventListener('click', async () => {
  const btn = $('btn-roi-save');
  if (ROI.enabled && ROI.points.length < 3) {
    toast('Add at least three corners before enabling the ROI.', 'err'); return;
  }
  btn.disabled = true; btn.textContent = 'Saving…';
  try {
    const saved = await api('/api/roi', { method: 'POST', body: {
      version: 1, width: 1600, height: 1200, enabled: ROI.enabled, points: ROI.points,
    } });
    ROI.points = saved.points; ROI.enabled = !!saved.enabled; ROI.loaded = true; ROI.dirty = false; renderROI();
    setText('roi-status', ROI.enabled
      ? `Saved and active · ${ROI.points.length} vertices. New scans will apply this mask before triangulation.`
      : ROI.points.length ? `Saved but inactive · ${ROI.points.length} vertices. Full-frame processing remains enabled.`
        : 'Saved · ROI disabled; full camera frame will be processed.');
    toast(ROI.enabled ? 'Camera ROI saved and enabled for future scans.' : 'Camera ROI saved; mask is disabled.');
  } catch (err) {
    toast(`Could not save camera ROI: ${err.message}`, 'err');
  } finally {
    btn.textContent = 'Save ROI';
    btn.disabled = false;
  }
});

async function refreshCameraDevices() {
  const btn = $('btn-refresh-cameras');
  if (btn) { btn.disabled = true; btn.textContent = 'Scanning…'; }
  try {
    const result = await api('/api/camera/devices');
    const select = $('camera-device');
    const previous = select.value;
    select.replaceChildren();
    for (const device of (result.devices || [])) {
      const option = document.createElement('option');
      option.value = JSON.stringify({ camera_id: device.camera_id, camera_backend: device.backend, name: device.name });
      option.textContent = device.label || `${device.name} — ${device.backend} (index ${device.camera_id})`;
      select.appendChild(option);
    }
    if (!select.options.length) {
      const option = document.createElement('option');
      option.value = ''; option.textContent = 'No cameras enumerated';
      select.appendChild(option);
      toast('No camera devices were enumerated. Check the USB connection and camera driver.', 'err');
    } else {
      const match = [...select.options].find((o) => o.value === previous);
      if (match) select.value = previous;
      else {
        const arducam = [...select.options].find((o) => /arducam|ov2311/i.test(o.textContent));
        select.value = (arducam || select.options[0]).value;
      }
      toast(`Found ${select.options.length} camera device/backend entries. Select the physical Arducam UVC device.`);
    }
  } catch (err) {
    $('camera-device').replaceChildren();
    const option = document.createElement('option');
    option.value = ''; option.textContent = 'Camera enumeration unavailable';
    $('camera-device').appendChild(option);
    toast(err.message, 'err');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Refresh list'; }
  }
}

async function probeCamera() {
  if (S.probing) return;
  S.probing = true;
  const btns = [$('btn-probe'), $('btn-probe-empty')];
  btns.forEach((b) => { b.disabled = true; b.textContent = 'Checking…'; });
  try {
    let camera = null;
    try { camera = JSON.parse($('camera-device').value || 'null'); } catch (e) { camera = null; }
    if (!camera) throw new Error('Refresh the camera list and select the physical Arducam UVC device by name.');
    const d = await api('/api/camera/probe', { method: 'POST', body: {
      camera_id: camera.camera_id, camera_backend: camera.camera_backend, camera_name: camera.name,
    } });
    S.cam = { state: 'ready', w: d.width, h: d.height, format: d.pixel_format, fps: d.measured_fps };
    await fetchPreviewOnce();
    previewLoop();
    setText('cam-res', `${d.width} × ${d.height}`);
    const measured = Number(d.measured_fps ?? d.fps ?? 0);
    const mode = `${d.pixel_format || 'unknown format'} · ${measured.toFixed(1)} FPS`;
    const light = ` mean ${d.brightness_mean}, p95 ${d.brightness_p95}/255`;
    if (d.width !== 1600 || d.height !== 1200) {
      toast(`${d.camera_name} returned ${d.width} × ${d.height}; the pipeline expects 1600 × 1200.`, 'err');
    } else if (d.pixel_format !== 'MJPG' || measured < 42.5) {
      toast(`${d.camera_name} is not ready for scanning: ${mode}. Required: MJPG and at least 42.5 measured FPS. Try the other camera backend and close other camera applications.${light}`, 'err');
    } else {
      toast(`${d.camera_name} is ready through ${d.camera_backend}: ${d.width} × ${d.height}, ${mode}.${light}`);
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
$('btn-refresh-cameras').addEventListener('click', refreshCameraDevices);
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
/* ---- library: saved scans (open, reprocess, build an arch, calibrate) ----------------- */
const LIB = { order: [], sessions: [] };
const VERDICT_LABEL = { good: 'Good', check: 'Usable', rescan: 'Rescan' };
const MOTION_LABEL = { markers: 'marker plate', planar: 'image tracking', flow: 'image shift', length: 'known length', speed: 'assumed speed' };
function fmtWhen(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const today = new Date().toDateString() === d.toDateString();
  const hm = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return today ? `today ${hm}` : `${d.toLocaleDateString([], { day: '2-digit', month: 'short' })} ${hm}`;
}
function renderLibrary() {
  const box = $('lib-list');
  if (!box) return;
  const busy = ['capturing', 'processing'].includes(S.snap && S.snap.phase);
  LIB.order = LIB.order.filter((n) => LIB.sessions.some((x) => x.name === n));
  box.innerHTML = LIB.sessions.length ? LIB.sessions.map((x) => {
    const k = LIB.order.indexOf(x.name);
    const bits = [fmtWhen(x.modified)];
    if (x.kind === 'calibration') bits.push('calibration recording');
    else if (x.kind === 'arch') bits.push('built arch');
    else {
      if (x.frames != null) bits.push(`${Math.round(x.frames / 50)} s`);
      bits.push(x.has_mesh ? (MOTION_LABEL[x.motion] || 'processed') : 'not processed');
    }
    const pill = x.kind === 'calibration' ? '<span class="pill">Calib</span>'
      : x.verdict ? `<span class="pill ${esc(x.verdict)}">${VERDICT_LABEL[x.verdict] || esc(x.verdict)}</span>` : '<span></span>';
    return `<div class="lib-item${k >= 0 ? ' on' : ''}" data-lib="${esc(x.name)}"><span class="ord">${k >= 0 ? k + 1 : ''}</span>`
      + `<span><div class="nm">${esc(x.name)}</div><div class="meta">${esc(bits.filter(Boolean).join(' · '))}</div></span>${pill}</div>`;
  }).join('') : '<div class="empty" style="border:0">No saved scans yet.</div>';
  const sel = LIB.order.map((n) => LIB.sessions.find((x) => x.name === n)).filter(Boolean);
  setText('lib-count', `${sel.length} selected`);
  const one = sel.length === 1 ? sel[0] : null;
  const calib = one && one.kind === 'calibration';
  const scans = sel.filter((x) => x.kind !== 'calibration');
  const allMarkers = scans.length > 0 && scans.every((x) => x.motion === 'markers');
  $('btn-lib-open').disabled = busy || !one || !one.has_mesh;
  const rp = $('btn-lib-reprocess');
  rp.textContent = calib ? 'Calibrate' : 'Reprocess';
  rp.disabled = busy || !one || one.kind === 'arch';
  const canMerge = scans.length === sel.length && scans.every((x) => x.has_mesh && x.kind === 'scan')
    && (scans.length >= 2 || (scans.length === 1 && allMarkers));
  $('btn-lib-merge').disabled = busy || !canMerge;
  let hint = 'Select one scan to open or reprocess it. Select several to build one arch from them.';
  if (calib) hint = 'Calibrate recomputes the calibration from this recording of the bare marker sheet.';
  else if (one && one.kind === 'arch') hint = 'A built arch can be opened and segmented from the Results tab.';
  else if (scans.length === 1 && allMarkers) hint = 'Build arch trims this marker-plate scan to the teeth, orients it and can segment it.';
  else if (scans.length >= 2 && allMarkers) hint = 'These scans share the marker plate: they are combined directly, without stitching.';
  else if (scans.length >= 2 && scans.some((x) => x.motion === 'markers')) hint = 'Marker-plate scans cannot be combined with scans made without the plate.';
  else if (scans.length >= 2) hint = 'Scans without the marker plate are stitched in the order selected; each must overlap the previous one.';
  setText('lib-hint', hint);
}
$('lib-list').addEventListener('click', (e) => {
  const it = e.target.closest('[data-lib]');
  if (!it) return;
  const n = it.dataset.lib, k = LIB.order.indexOf(n);
  if (k >= 0) LIB.order.splice(k, 1); else LIB.order.push(n);
  renderLibrary();
});
function freshView() {
  V.reset?.();
  Object.assign(S, { origin: null, userMoved: false, needFit: true, loadedKey: null, failKey: null, view: 'auto', lastGood: null });
}
async function refreshSessions() {
  try {
    const { sessions } = await api('/api/sessions');
    LIB.sessions = sessions;
  } catch (err) { /* keep the previous list; the status bar reports connection problems */ }
  renderLibrary();
}
async function openResult() {
  const name = LIB.order[0];
  if (!name) return;
  S.clientAlert = null;
  try {
    await api('/api/results/open', { method: 'POST', body: { name } });
    freshView();
    switchTab('scan');
  } catch (err) { showClientAlert('Could not open the scan', err.message); toast(err.message, 'err'); }
  await pollNow();
}
async function reprocessSession() {
  const name = LIB.order[0];
  const item = LIB.sessions.find((x) => x.name === name);
  if (!item) { toast('Select a saved scan first.', 'err'); return; }
  if (item.kind === 'calibration') return calibrateFrom(name);
  const recon = reconSettings();
  if (!validRecon(recon)) { toast('Check the reconstruction settings in Setup → Advanced.', 'err'); return; }
  S.clientAlert = null;
  try {
    await api('/api/process/start', { method: 'POST', body: {
      ...recon, session_name: name, depth_min_mm: Number($('depth-min').value), depth_max_mm: Number($('depth-max').value),
      speed_mm_s: Number($('speed').value), jaw: $('jaw').value, mesh_method: $('mesh-method').value,
      auto_segment: false, models_dir: $('models-dir').value.trim() || null,
      use_calibration: $('use-calibration').checked,
    } });
    freshView();
    switchTab('scan');
    toast(`Reprocessing ${name}…`);
  } catch (err) { showClientAlert('Could not reprocess the scan', err.message); toast(err.message, 'err'); }
  await pollNow();
}
async function buildArch() {
  if (!LIB.order.length) return;
  const name = $('stitch-name').value.trim();
  if (name && !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)) { toast('Arch name: letters, numbers, “_” and “-” only.', 'err'); return; }
  S.clientAlert = null;
  try {
    const body = { passes: LIB.order.slice(), keep_base: $('stitch-keep-base').checked,
      auto_segment: $('stitch-segment').checked, jaw: $('jaw').value, models_dir: $('models-dir').value.trim() || null };
    if (name) body.name = name;
    const r = await api('/api/stitch/start', { method: 'POST', body });
    freshView();
    switchTab('scan');
    toast(`Building ${r.name} from ${LIB.order.length} scan${LIB.order.length === 1 ? '' : 's'}…`);
  } catch (err) { showClientAlert('Could not build the arch', err.message); toast(err.message, 'err'); }
  await pollNow();
}
$('btn-lib-refresh').addEventListener('click', refreshSessions);
$('btn-lib-open').addEventListener('click', openResult);
$('btn-lib-reprocess').addEventListener('click', reprocessSession);
$('btn-lib-merge').addEventListener('click', buildArch);

/* ---- calibration ---------------------------------------------------------------------- */
const CAL = { status: null };
function barLength() {
  const v = Number($('bar-length').value);
  return Number.isFinite(v) && v >= 90 && v <= 110 ? v : null;
}
function renderCalibration() {
  const c = CAL.status;
  if (!c) return;
  const ok = !!c.calibrated;
  const q = ok ? (c.quality === 'good' ? 'ok' : 'warn') : '';
  $('cal-pill').className = `pill ${q}`;
  setText('cal-pill', ok ? (c.quality === 'good' ? 'Calibrated' : 'Check') : 'Nominal');
  setText('cal-title', ok ? 'Calibrated' : 'Not calibrated');
  setText('sum-calibration', ok ? `Calibrated ${c.created ? fmtWhen(c.created) : ''}`.trim() : 'Not calibrated');
  $('setup-dot').hidden = ok;
  if (c.error) { setText('cal-detail', c.error); return; }
  if (ok) {
    const parts = [`Focal length ${Math.round(c.focal_px)} px`];
    if (c.focal_change_percent != null) parts[0] += ` (${c.focal_change_percent > 0 ? '+' : ''}${Number(c.focal_change_percent).toFixed(1)}% vs nominal)`;
    if (c.camera_rms_px != null) parts.push(`marker fit ${Number(c.camera_rms_px).toFixed(2)} px`);
    if (c.sheet_residual_mm != null) parts.push(`sheet residual ${Number(c.sheet_residual_mm).toFixed(2)} mm`);
    if (c.mirrored_optics) parts.push('mirror in the optical path detected');
    setText('cal-detail', `${parts.join(' · ')}.${(c.notes || []).length ? ' ' + c.notes.join(' ') : ''} Recalibrate only if the camera, lens or laser is moved.`);
    if (c.board && c.board.scale && document.activeElement !== $('bar-length')) $('bar-length').value = (c.board.scale * 100).toFixed(1).replace(/\.0$/, '');
  } else {
    setText('cal-detail', 'The scanner is using nominal lens and laser values. Calibrate once with the bare marker sheet; redo it only if the camera, lens or laser is moved.');
  }
}
async function refreshCalibration() {
  try { CAL.status = await api('/api/calibration'); renderCalibration(); } catch (e) { /* shown on next refresh */ }
}
async function recordCalibration() {
  const bar = barLength();
  if (bar === null) { toast('Enter the measured length of the printed 100 mm bar (90 to 110 mm).', 'err'); return; }
  const payload = readForm();
  if (!payload) return;
  delete payload.session_name;
  Object.assign(payload, { purpose: 'calibration', bar_length_mm: bar, frames: 1500, auto_segment: false, calibration_file: null });
  S.clientAlert = null;
  try {
    await api('/api/capture/start', { method: 'POST', body: payload });
    freshView();
    switchTab('scan');
    toast('Recording calibration: slide, lift and tilt the bare marker sheet for 30 seconds.', 'info', 12000);
  } catch (err) { showClientAlert('Could not start the calibration recording', err.message); toast(err.message, 'err'); }
  await pollNow();
}
async function calibrateFrom(name) {
  const bar = barLength();
  if (bar === null) { toast('Enter the measured bar length in Setup → Calibration first.', 'err'); openSetup('acc-calibration'); return; }
  S.clientAlert = null;
  try {
    await api('/api/calibration/start', { method: 'POST', body: { session_name: name, bar_length_mm: bar } });
    switchTab('scan');
    toast(`Calibrating from ${name}…`);
  } catch (err) { showClientAlert('Could not start the calibration', err.message); toast(err.message, 'err'); }
  await pollNow();
}
$('btn-cal-record').addEventListener('click', recordCalibration);

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
  ready: 'System ready', capturing: 'Recording', captured: 'Recording saved', reconstructing: 'Building the 3D surface',
  meshing: 'Building the surface mesh', mesh_ready: 'Surface mesh ready', reconstructed: 'Reconstruction complete',
  segmenting: 'Segmenting teeth', stitching: 'Building the arch', quality_error: 'Needs review', complete: 'Processing complete', segmentation_error: 'Segmentation needs review', error: 'Stopped with an error',
  calibrating: 'Calibrating', calibrated: 'Calibration saved',
};

function stepStates(snap) {
  const { phase, stage } = snap;
  const out = snap.outputs || {};
  const hasMesh = !!out.mesh;
  const recStages = ['reconstructing', 'meshing', 'mesh_ready', 'calibrating', 'stitching'];
  const failedAt = S.lastGood;
  const errored = phase === 'error';

  let cap;
  if (phase === 'capturing') cap = ['active', 'Recording'];
  else if (errored && (failedAt === 'capturing' || !failedAt)) cap = ['error', 'Failed'];
  else if (phase === 'idle') cap = ['', 'Idle'];
  else cap = ['done', `${fmtInt(snap.frame)} frames`];

  let rec;
  if (recStages.includes(stage)) rec = ['active', 'Running'];
  else if (['reconstructed', 'segmenting', 'complete', 'segmentation_error', 'calibrated'].includes(stage) || hasMesh) rec = ['done', 'Done'];
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
  else if (S.cam.state === 'ready') { cls = 'ok'; txt = S.cam.w ? `Camera · ${S.cam.w}×${S.cam.h}` : 'Camera · live'; }
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
  setText('ro-frames-sub', capturing ? (S.capStart !== null ? fmtClock(performance.now() - S.capStart) : '--:--') : snap.frame ? 'saved' : 'Not recording');

  const fps = capturing ? S.fps : S.capAvg;
  const low = capturing && fps != null && snap.frame > 60 && fps < TARGET_FPS * 0.9;
  $('ro-fps').firstChild.nodeValue = fps != null ? fps.toFixed(1) : '—';
  const sub = $('ro-fps-sub');
  sub.className = `sub ${low ? 'warn' : ''}`;
  sub.textContent = low ? `below ${TARGET_FPS} target`
    : capturing ? (fps == null ? 'measuring…' : `target ${TARGET_FPS}`)
    : S.capAvg != null ? 'average' : `target ${TARGET_FPS}`;

  const live = snap.model_asset === 'preview' && snap.phase !== 'complete';
  setText('ro-points-label', live ? 'Live points' : 'Fused points');
  setText('ro-points', snap.point_count ? fmtInt(snap.point_count) : '—');
  setText('ro-points-sub', snap.point_count ? (live ? 'live preview' : 'fused') : 'No points yet');
  setText('ro-faces', snap.mesh_faces ? fmtInt(snap.mesh_faces) : '—');
  setText('ro-faces-sub', snap.mesh_vertices ? `${fmtInt(snap.mesh_vertices)} vertices` : 'No surface yet');
  $('ro-path').firstChild.nodeValue = snap.trajectory_mm ? Number(snap.trajectory_mm).toFixed(1) : '—';
  const TR = { markers: 'marker plate', planar: 'image tracking', flow: 'image shift', length: 'known length', speed: 'assumed speed' };
  setText('ro-path-sub', snap.trajectory_mm ? (TR[snap.translation_source] || 'assumed speed') : 'No scan yet');
}

function renderCameraMeta(snap) {
  const capturing = snap.phase === 'capturing';
  $('cam-hud').hidden = !(S.camShown || snap.frame > 0);
  $('hud-rec').firstElementChild.className = `dot ${capturing ? 'rec' : ''}`;
  setText('hud-rec-t', capturing ? 'REC' : snap.camera_live ? 'Live' : snap.frame ? 'Captured' : 'Standby');
  setText('hud-laser', snap.camera_live && !capturing ? 'Live preview · laser off'
    : `Last frame · laser ${snap.frame ? (snap.laser_on ? 'ON' : 'OFF') : '—'}`);
  setText('cam-meta-r', `Frame ${snap.frame ? fmtInt(snap.frame) : '—'} / ${snap.frames_total ? fmtInt(snap.frames_total) : '—'}`);
  setText('cam-meta-l', capturing ? 'Streaming from scanner'
    : snap.camera_live ? 'Live camera preview'
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
  let count = '';
  if (stage === 'capturing' && total) count = `${fmtInt(snap.frame)} / ${fmtInt(total)} · `;
  else if ((stage === 'reconstructing' || stage === 'meshing') && total) count = `${fmtInt(snap.processed_frames || 0)} / ${fmtInt(total)} · `;
  setText('prog-count', count);
  const msg = snap.error ? `${snap.message} ${snap.error}` : (snap.message || '');
  setText('status-msg', msg);
  setText('foot-msg', snap.error ? snap.error : (snap.message || 'Ready'));
}

function hintFor(error, where) {
  const e = String(error || '');
  if (/queue overflow/i.test(e)) return 'The PC capture loop fell behind the ESP32 strobe stream. IOSCAN now stores every JPEG directly and samples only selected OFF/ON pairs for live preview; verify the measured capture rate and host queue depth before retrying.';
  if (/sequence gap/i.test(e)) return 'One or more physical camera STROBE events did not produce a packet. Check ESP32 ring-overflow / stale-IMU warnings, USB serial integrity, and ensure no other app is reading the COM port.';
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
    ? 'Points appear as soon as the laser line is found on the model.'
    : 'The scan appears here while it is recorded and processed.');
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
    : '<div class="empty">Outputs appear here once processing finishes.</div>';

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

const READY_ORDER = ['markers', 'line', 'light', 'calibration'];
const READY_PLACEHOLDER = { markers: 'Marker plate', line: 'Laser line', light: 'Background light', calibration: 'Calibration' };
function renderReady(snap) {
  const rc = snap.ready_check;
  const pill = $('ready-pill');
  if (!rc) {
    pill.className = 'pill';
    setText('ready-pill', snap.camera_live || snap.phase === 'capturing' ? 'Checking…' : 'Waiting for camera');
    if (!snap.camera_live && snap.phase !== 'capturing') {
      setText('ready-advice', S.cam.state === 'ready' ? 'Waiting for the first frames…' : 'Check the camera to start the ready check.');
    }
    return;
  }
  pill.className = `pill ${rc.state}`;
  setText('ready-pill', rc.title);
  const by = Object.fromEntries((rc.items || []).map((i) => [i.key, i]));
  const html = READY_ORDER.map((k) => {
    const it = by[k] || { state: 'idle', label: READY_PLACEHOLDER[k], value: '—' };
    return `<li class="${esc(it.state)}"${k === 'calibration' && it.state !== 'ok' ? ' data-open="acc-calibration" style="cursor:pointer" title="Open calibration"' : ''}>`
      + `<span class="dot ${esc(it.state)}"></span><span class="rk">${esc(it.label)}</span><span class="rv">${esc(it.value)}</span></li>`;
  }).join('');
  const box = $('ready-items');
  if (box.dataset.html !== html) { box.innerHTML = html; box.dataset.html = html; }
  setText('ready-advice', rc.advice || (rc.state === 'ok' ? 'Everything needed for a dependable scan is in place.' : ''));
}
$('ready-items').addEventListener('click', (e) => { const li = e.target.closest('[data-open]'); if (li) openSetup(li.dataset.open); });

function renderVerdict(snap) {
  const v = snap.verdict;
  const busy = ['capturing', 'processing'].includes(snap.phase);
  $('sec-verdict').hidden = !v || busy;
  if (!v || busy) return;
  $('verdict').className = `verdict ${v.status}`;
  setText('verdict-title', v.title);
  $('verdict-pill').className = `pill ${v.status}`;
  setText('verdict-pill', VERDICT_LABEL[v.status] || v.status);
  const reasons = v.reasons || [];
  const more = reasons.slice(2);
  const html = reasons.slice(0, 2).map((r) => `<li>${esc(r)}</li>`).join('')
    + (more.length ? `<li style="list-style:none;margin-left:-16px"><details class="inline"><summary>${more.length} more</summary>`
      + `<ul style="margin-top:6px">${more.map((r) => `<li>${esc(r)}</li>`).join('')}</ul></details></li>` : '');
  const ul = $('verdict-reasons');
  if (ul.dataset.html !== html) { ul.innerHTML = html; ul.dataset.html = html; }
  ul.hidden = reasons.length === 0;
  const m = v.metrics || {};
  setText('verdict-metrics', [m.area_mm2 != null ? `${Math.round(m.area_mm2)} mm² surface` : null,
    m.points_per_profile != null ? `${Math.round(m.points_per_profile)} points per profile` : null,
    m.motion_source ? `motion: ${MOTION_LABEL[m.motion_source] || m.motion_source}` : null].filter(Boolean).join(' · '));
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

  if (S.lastPhase !== snap.phase && ['complete', 'complete_with_warnings', 'captured', 'error'].includes(snap.phase)) { refreshSessions(); refreshCalibration(); }
  S.lastPhase = snap.phase;
  updateFps(snap);
  S.wantPreview = snap.phase === 'capturing';
  if (snap.camera_live && S.cam.state !== 'ready') S.cam = { ...S.cam, state: 'ready' };
  if (S.wantPreview || snap.camera_live) previewLoop();

  renderChips(snap);
  renderWorkflow(snap);
  renderReadouts(snap);
  renderCameraMeta(snap);
  renderReady(snap);
  renderVerdict(snap);
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
  updateReconFields();
  await loadHardwareGeometry();
  applyTheme(document.documentElement.getAttribute('data-theme') || 'light');
  updateSetupSummaries();
  poll();
  refreshCameraDevices().then(updateSetupSummaries);
  refreshSessions();
  refreshCalibration();
  loadROI();
  await V.init();
  if (V.ok) { V.applyTheme(); if (S.snap) maybeLoadModel(S.snap); }
})();