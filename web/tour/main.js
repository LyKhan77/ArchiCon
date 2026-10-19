import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { calibrationDocument, validateCalibrationDocument } from './calibration.js';

// Input positions, camera matrices, and floorplan coordinates already share Y-up axes.
// Preserve them here; calibration changes displayed units only.
const data = JSON.parse(document.getElementById('tour-data').textContent);
const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const state = { component: null, index: 0, tab: '3d', touring: false, playing: false, nextAt: 0, gap: false, token: 0, evidence: true };
const drafts = new Map();
const models = new Map();
const loader = new GLTFLoader();
let renderer, camera, controls, scene, currentModel, clipPlane, planSvg;
let planTransform;

function draft() {
  const id = state.component.id;
  if (!drafts.has(id)) drafts.set(id, { scale: null, points: [], reference: null });
  return drafts.get(id);
}
function text(id, value) { $(id).textContent = value; }
function number(value) { return new Intl.NumberFormat('id-ID', { maximumFractionDigits: 2 }).format(value); }
function length(value) { return draft().scale ? `≈ ${number(value * draft().scale)} m` : `${number(value)} u`; }
function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
function fail(error) {
  $('loading').hidden = true;
  $('error').hidden = false;
  text('error', `Tampilan gagal dimuat. ${error?.message || error}. Denah 2D tetap dapat dipilih jika data tersedia.`);
  console.error(error);
}
function svgNode(name, attrs = {}, label) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (label !== undefined) node.textContent = label;
  return node;
}

function init3d() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#202020');
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  $('canvas-host').append(renderer.domElement);
  renderer.domElement.setAttribute('aria-label', 'Model 3D; kontrol putar dan zoom menggunakan mouse atau sentuhan');
  renderer.domElement.setAttribute('tabindex', '0');
  camera = new THREE.PerspectiveCamera(65, 1, 0.01, 1000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);
  let drag = null;
  const rotateInPlace = (dx, dy) => {
    if (!state.touring) return;
    stop();
    const angles = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    angles.y -= dx * 0.004;
    angles.x = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, angles.x - dy * 0.004));
    camera.quaternion.setFromEuler(angles);
    controls.target.copy(camera.position).add(new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion));
    text('view-mode', 'Kamera rekaman / lihat sekitar');
  };
  renderer.domElement.addEventListener('pointerdown', (event) => {
    if (!state.touring || event.button !== 0) return;
    drag = { x: event.clientX, y: event.clientY, pointerId: event.pointerId };
    renderer.domElement.setPointerCapture(event.pointerId);
  });
  renderer.domElement.addEventListener('pointermove', (event) => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    rotateInPlace(event.clientX - drag.x, event.clientY - drag.y);
    drag.x = event.clientX; drag.y = event.clientY;
  });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) renderer.domElement.addEventListener(name, () => { drag = null; });
  renderer.domElement.addEventListener('keydown', (event) => {
    if (!state.touring || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    rotateInPlace(event.key === 'ArrowLeft' ? -12 : event.key === 'ArrowRight' ? 12 : 0, event.key === 'ArrowUp' ? -12 : event.key === 'ArrowDown' ? 12 : 0);
  });
  new ResizeObserver(resize).observe($('canvas-host'));
  resize();
}
function resize() {
  if (!renderer) return;
  const { clientWidth: width, clientHeight: height } = $('canvas-host');
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
function decodeGLB(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}
async function loadModel(component) {
  if (!models.has(component.id)) {
    const promise = loader.parseAsync(decodeGLB(component.embedded_glb), '').then((gltf) => {
      gltf.scene.traverse((object) => {
        if (!object.isMesh) return;
        const old = Array.isArray(object.material) ? object.material : [object.material];
        object.material = new THREE.MeshBasicMaterial({
          vertexColors: !!object.geometry.getAttribute('color'),
          color: object.geometry.getAttribute('color') ? 0xffffff : 0xb7b9ae,
          side: THREE.DoubleSide,
        });
        old.forEach((material) => material?.dispose());
      });
      return gltf.scene;
    });
    models.set(component.id, promise);
  }
  return models.get(component.id);
}
async function selectComponent(id) {
  const token = ++state.token;
  stop();
  state.component = data.components.find((item) => item.id === id);
  $('component').value = id;
  for (const button of $('component-list').querySelectorAll('[data-component]')) {
    button.setAttribute('aria-pressed', button.dataset.component === id);
  }
  window.__TOUR_READY__ = false;
  state.index = 0;
  state.gap = false;
  state.touring = false;
  text('component-title', state.component.name);
  $('frame').max = Math.max(0, state.component.cameras.length - 1);
  $('frame').disabled = !state.component.cameras.length;
  $('play').disabled = state.component.cameras.length < 2 || !renderer;
  $('real-length').value = '';
  $('roof').value = 65;
  drawPlan();
  updateMeasurement();
  updateFrameLabel();
  if (!renderer) return;
  $('loading').hidden = false;
  try {
    const model = await loadModel(state.component);
    if (token !== state.token) return;
    if (currentModel) scene.remove(currentModel);
    currentModel = model;
    scene.add(model);
    overview();
    $('loading').hidden = true;
    window.__TOUR_READY__ = true;
  } catch (error) { if (token === state.token) fail(error); }
}
function overview() {
  stop();
  state.touring = false;
  state.gap = false;
  const [min, max] = state.component.bounds;
  const center = new THREE.Vector3(...min).add(new THREE.Vector3(...max)).multiplyScalar(0.5);
  const diagonal = Math.max(0.01, new THREE.Vector3(...max).distanceTo(new THREE.Vector3(...min)));
  if (camera) {
    camera.near = Math.max(diagonal / 10000, 0.0001);
    camera.far = diagonal * 100;
    camera.fov = 65;
    camera.up.set(0, 1, 0);
    const towardsCamera = new THREE.Vector3(0.75, 0.7, 0.9).normalize();
    const right = new THREE.Vector3().crossVectors(camera.up, towardsCamera).normalize();
    const up = new THREE.Vector3().crossVectors(towardsCamera, right).normalize();
    const tanVertical = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const tanHorizontal = tanVertical * camera.aspect;
    let fitDistance = diagonal * 0.1;
    for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) {
      const corner = new THREE.Vector3(x, y, z).sub(center);
      const depth = corner.dot(towardsCamera);
      fitDistance = Math.max(fitDistance, Math.abs(corner.dot(right)) / tanHorizontal + depth, Math.abs(corner.dot(up)) / tanVertical + depth);
    }
    camera.position.copy(center).addScaledVector(towardsCamera, fitDistance * 1.15);
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.enabled = true;
    controls.update();
    updateRoof();
  }
  text('view-mode', 'Ikhtisar / orbit');
  text('gap-status', 'Pilih kamera untuk melihat sudut rekaman.');
  $('gap-status').classList.remove('has-gap');
  $('roof').disabled = state.tab !== '3d' || !renderer;
  $('orbit-note').hidden = false;
  text('orbit-note', 'Geser: putar · roda / cubit: zoom · klik kanan: geser');
  if (renderer) renderer.domElement.style.cursor = 'grab';
  setPlayLabel();
}
function updateRoof() {
  if (!renderer) return;
  const percent = Number($('roof').value);
  text('roof-value', `${percent}%`);
  const [min, max] = state.component.bounds;
  clipPlane.constant = min[1] + (max[1] - min[1]) * percent / 100;
  renderer.clippingPlanes = state.touring || percent === 100 ? [] : [clipPlane];
}
function updateFrameLabel() {
  const frame = state.component.cameras[state.index];
  $('frame').value = state.index;
  text('frame-label', frame ? `${state.index + 1} / ${state.component.cameras.length} · ${frame.frame_id}` : 'tidak tersedia');
  text('time-label', frame ? `${number(frame.time)} s` : '—');
}
function showFrame(index, clearGap = true) {
  state.index = Math.max(0, Math.min(index, state.component.cameras.length - 1));
  const frame = state.component.cameras[state.index];
  if (!frame || !camera) return;
  state.touring = true;
  if (clearGap) state.gap = false;
  controls.enabled = false;
  // Matrix4.set takes row-major values. fromArray would transpose this contract.
  const matrix = new THREE.Matrix4().set(...frame.matrix.flat());
  matrix.decompose(camera.position, camera.quaternion, new THREE.Vector3());
  camera.up.set(0, 1, 0).applyQuaternion(camera.quaternion);
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).add(forward);
  const suppliedFov = Number(frame.vertical_fov_degrees);
  const calibratedFov = Number.isFinite(suppliedFov) && suppliedFov > 1 && suppliedFov < 179;
  camera.fov = calibratedFov ? suppliedFov : 65;
  camera.updateProjectionMatrix();
  renderer.clippingPlanes = [];
  $('roof').disabled = true;
  $('orbit-note').hidden = false;
  text('orbit-note', 'Geser / tombol panah: lihat sekitar · slider: reset pose rekaman');
  renderer.domElement.style.cursor = 'grab';
  text('view-mode', `Kamera rekaman / FOV vertikal ${calibratedFov ? '' : '≈'}${number(camera.fov)}°`);
  const previous = state.component.cameras[state.index - 1];
  const gap = previous && frame.time - previous.time > 2;
  text('gap-status', gap ? `Sesudah jeda sumber ${number(frame.time - previous.time)} s. Lintasan antar-jeda tidak diinterpolasi.` : calibratedFov ? 'FOV dari kalibrasi kamera; rasio layar berbeda dari rekaman.' : 'Perspektif perkiraan 65°; intrinsics kamera tidak tersedia. Bidang horizontal mengikuti viewport.');
  $('gap-status').classList.toggle('has-gap', !!gap);
  updateFrameLabel();
  updatePlanSelection();
  setPlayLabel();
}
function setPlayLabel() {
  $('play').replaceChildren(document.createTextNode(state.playing ? 'Ⅱ ' : '▶ '));
  const span = document.createElement('span');
  span.textContent = state.playing ? 'Jeda' : state.gap ? 'Lewati jeda & putar' : 'Putar tur';
  $('play').append(span);
  $('play').setAttribute('aria-label', span.textContent);
}
function stop() { state.playing = false; setPlayLabel(); }
function play() {
  if (state.playing) return stop();
  if (!state.component.cameras.length || !camera) return;
  if (state.gap) showFrame(state.index + 1);
  else if (state.index >= state.component.cameras.length - 1) showFrame(0);
  else showFrame(state.index);
  state.playing = true;
  const next = state.component.cameras[state.index + 1];
  state.nextAt = performance.now() + (next ? Math.max(0, next.time - state.component.cameras[state.index].time) * 1000 : 0);
  setPlayLabel();
}
function animate(now) {
  requestAnimationFrame(animate);
  if (state.playing && now >= state.nextAt) {
    const frames = state.component.cameras;
    const next = frames[state.index + 1];
    if (!next) stop();
    else if (next.time - frames[state.index].time > 2) {
      state.gap = true;
      stop();
      text('gap-status', `Tur berhenti: jeda sumber ${number(next.time - frames[state.index].time)} s. Tekan “Lewati jeda & putar” untuk lanjut.`);
      $('gap-status').classList.add('has-gap');
    } else {
      showFrame(state.index + 1);
      const following = frames[state.index + 1];
      state.nextAt = now + (following ? Math.max(0, following.time - next.time) * 1000 : 0);
    }
  }
  if (renderer && camera && state.tab === '3d') {
    if (!state.touring) controls.update();
    renderer.render(scene, camera);
  }
}
function setTab(tab) {
  state.tab = tab;
  if (tab === 'plan') stop();
  for (const name of ['3d', 'plan']) {
    $(`tab-${name}`).classList.toggle('active', name === tab);
    $(`tab-${name}`).setAttribute('aria-selected', name === tab);
    $(`tab-${name}`).tabIndex = name === tab ? 0 : -1;
    $(`panel-${name}`).hidden = name !== tab;
  }
  $('tour-controls').hidden = tab !== '3d';
  $('measure-controls').hidden = tab !== 'plan';
  $('calibration-files').hidden = tab !== 'plan';
  $('roof').disabled = tab !== '3d' || state.touring || !renderer;
  if (tab === '3d') resize();
}

function drawPlan() {
  const component = state.component;
  const plan = component.floorplan;
  const [min, max] = plan.bounds;
  const width = Math.max(max[0] - min[0], 0.01), height = Math.max(max[1] - min[1], 0.01);
  const displayMin = [...min], displayMax = [...max];
  for (const frame of component.cameras) {
    displayMin[0] = Math.min(displayMin[0], frame.position[0]);
    displayMin[1] = Math.min(displayMin[1], frame.position[2]);
    displayMax[0] = Math.max(displayMax[0], frame.position[0]);
    displayMax[1] = Math.max(displayMax[1], frame.position[2]);
  }
  const unit = Math.max(width, height), margin = unit * 0.13;
  planTransform = { width, height, unit, margin, min, max };
  planSvg = svgNode('svg', { viewBox: `${displayMin[0] - margin} ${displayMin[1] - margin} ${displayMax[0] - displayMin[0] + 2 * margin} ${displayMax[1] - displayMin[1] + 2 * margin}`, role: 'img', 'aria-label': `Denah ${component.name}; klik dua titik pada geometri untuk mengukur` });
  planSvg.append(svgNode('title', {}, `Denah rekonstruksi ${component.name}`));
  planSvg.append(svgNode('desc', {}, 'Proyeksi X/Z dalam koordinat asli. Segmen adalah irisan geometri; bukan denah bangunan tersertifikasi. Dimensi keseluruhan mencakup outlier.'));
  const style = svgNode('style');
  style.textContent = '.wall{fill:none;stroke:#393939;stroke-width:1.1;vector-effect:non-scaling-stroke}.path{fill:none;stroke:#0f62fe;stroke-width:1.3;stroke-dasharray:4 3;vector-effect:non-scaling-stroke;opacity:.7}.camera{fill:#0f62fe;stroke:#f4f4f4;stroke-width:1;vector-effect:non-scaling-stroke;cursor:pointer}.camera.selected{fill:#161616;stroke:#0f62fe;stroke-width:2}.dimension{fill:none;stroke:#8d8d8d;stroke-width:.75;vector-effect:non-scaling-stroke}.dimension-text{fill:#525252;stroke:none;letter-spacing:normal;font-family:Consolas,monospace;text-anchor:middle}.measurement-line{stroke:#0f62fe;stroke-width:2;vector-effect:non-scaling-stroke}.measurement-point{fill:#0f62fe;stroke:#fff;stroke-width:1.5;vector-effect:non-scaling-stroke}.measurement-text{fill:#0f62fe;letter-spacing:normal;paint-order:stroke;stroke:#f4f4f4;stroke-width:3px;stroke-linejoin:round;font-family:Consolas,monospace;text-anchor:middle}';
  planSvg.append(style);
  $('evidence-toggle').disabled = !plan.embedded_image;
  if (plan.embedded_image) {
    const [imageMin, imageMax] = plan.image_bounds || plan.bounds;
    const image = svgNode('image', { id: 'plan-evidence', href: plan.embedded_image, x: imageMin[0], y: imageMin[1], width: imageMax[0] - imageMin[0], height: imageMax[1] - imageMin[1], preserveAspectRatio: 'none', opacity: 0.48, style: `mix-blend-mode:multiply;pointer-events:none;display:${state.evidence ? 'inline' : 'none'}` });
    image.append(svgNode('title', {}, 'Bukti irisan geometri yang belum diklasifikasikan; dapat mencakup furnitur.'));
    planSvg.append(image);
  }
  const walls = svgNode('path', { class: 'wall', d: plan.segments.map(([a,b]) => `M${a[0]},${a[1]}L${b[0]},${b[1]}`).join('') });
  planSvg.append(walls);
  let path = '';
  let previous;
  for (const point of plan.path || []) {
    const gap = previous && point.time - previous.time > 2;
    path += `${!previous || point.break_before || gap ? 'M' : 'L'}${point.point[0]},${point.point[1]}`;
    previous = point;
  }
  planSvg.append(svgNode('path', { class: 'path', d: path }));
  const cameraGroup = svgNode('g', { 'aria-label': 'Kamera rekaman' });
  component.cameras.forEach((frame, index) => {
    const circle = svgNode('circle', { class: 'camera', cx: frame.position[0], cy: frame.position[2], r: unit * 0.0035, tabindex: 0, role: 'button', 'aria-label': `Kamera ${frame.frame_id}, ${number(frame.time)} detik`, 'data-camera': index });
    circle.append(svgNode('title', {}, `Kamera ${frame.frame_id} · ${number(frame.time)} s`));
    const activate = (event) => { event.stopPropagation(); stop(); showFrame(index); setTab('3d'); };
    circle.addEventListener('click', activate);
    circle.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); activate(event); } });
    cameraGroup.append(circle);
  });
  planSvg.append(cameraGroup);
  planSvg.append(svgNode('g', { id: 'plan-dimensions' }), svgNode('g', { id: 'plan-measurement' }));
  planSvg.addEventListener('click', (event) => {
    if (event.target.closest('[data-camera]')) return;
    const matrix = planSvg.getScreenCTM();
    if (!matrix) return;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    if (point.x < min[0] || point.x > max[0] || point.y < min[1] || point.y > max[1]) return;
    const selected = draft().points;
    if (selected.length === 2) selected.length = 0;
    selected.push([point.x, point.y]);
    updateMeasurement();
  });
  $('plan-host').replaceChildren(planSvg);
  updateDimensions();
  updatePlanSelection();
}
function updatePlanSelection() {
  planSvg?.querySelectorAll('[data-camera]').forEach((node) => node.classList.toggle('selected', state.touring && Number(node.dataset.camera) === state.index));
}
function updateDimensions() {
  if (!planSvg) return;
  const { width, height, unit, margin, min, max } = planTransform;
  const group = planSvg.querySelector('#plan-dimensions');
  group.replaceChildren();
  const offset = margin * 0.42, tick = unit * 0.012, fontSize = unit * 0.024;
  const z = max[1] + offset, x = max[0] + offset;
  group.append(svgNode('path', { class: 'dimension', d: `M${min[0]},${z}H${max[0]}M${min[0]},${z-tick}v${2*tick}M${max[0]},${z-tick}v${2*tick}M${x},${min[1]}V${max[1]}M${x-tick},${min[1]}h${2*tick}M${x-tick},${max[1]}h${2*tick}` }));
  group.append(svgNode('text', { class: 'dimension-text', x: (min[0]+max[0])/2, y: z+fontSize*1.6, 'font-size': fontSize }, `X · ${length(width)}`));
  group.append(svgNode('text', { class: 'dimension-text', transform: `translate(${x+fontSize*1.6} ${(min[1]+max[1])/2}) rotate(-90)`, 'font-size': fontSize }, `Z · ${length(height)}`));
  group.append(svgNode('text', { class: 'dimension-text', x: (min[0]+max[0])/2, y: min[1]-margin*.5, 'font-size': fontSize*.8 }, 'BENTANG GEOMETRI · TERMASUK OUTLIER'));
  text('bounds-summary', `Bentang geometri X ${length(width)} × Z ${length(height)} · bukan dimensi ruangan`);
}
function updateMeasurement() {
  const current = draft();
  const points = current.points;
  const value = points.length === 2 ? distance(...points) : null;
  text('measurement', value === null ? '—' : length(value));
  text('measurement-detail', points.length === 1 ? 'Titik awal dipilih · pilih titik akhir' : points.length === 2 ? 'Jarak lurus pada proyeksi X/Z' : 'Pilih titik awal dan akhir');
  text('measure-help', points.length === 2 ? 'Klik lagi untuk memilih segmen baru.' : points.length === 1 ? 'Klik titik akhir pada denah.' : 'Klik dua titik pada denah untuk mengukur.');
  $('calibrate').disabled = !value;
  text('calibration-state', current.scale ? `Draft komponen ini: 1 u ≈ ${number(current.scale)} m. Acuan ${number(current.reference.meters)} m; bukan hasil survei.` : 'Belum ada acuan ukuran nyata. Satuan u adalah satuan rekonstruksi.');
  text('scale-state', current.scale ? 'Skala draft terkalibrasi · ≈m' : 'Skala belum terkalibrasi');
  text('footer-note', current.scale ? 'Kalibrasi draft · ukuran ruangan belum terverifikasi' : 'Satuan u · ukuran ruangan belum terverifikasi');
  const group = planSvg?.querySelector('#plan-measurement');
  if (group) {
    group.replaceChildren();
    const unit = planTransform.unit;
    if (points.length === 2) {
      group.append(svgNode('line', { class: 'measurement-line', x1: points[0][0], y1: points[0][1], x2: points[1][0], y2: points[1][1] }));
      group.append(svgNode('text', { class: 'measurement-text', x: (points[0][0]+points[1][0])/2, y: (points[0][1]+points[1][1])/2-unit*.013, 'font-size': unit*.02 }, length(value)));
    }
    for (const point of points) group.append(svgNode('circle', { class: 'measurement-point', cx: point[0], cy: point[1], r: unit*.006 }));
  }
  updateDimensions();
}
function exportSvg() {
  const exported = planSvg.cloneNode(true);
  exported.removeAttribute('xmlns');
  exported.setAttribute('width', '1400');
  exported.setAttribute('height', '1000');
  exported.removeAttribute('role');
  exported.querySelectorAll('[tabindex]').forEach((node) => { node.removeAttribute('tabindex'); node.removeAttribute('role'); });
  const calibration = calibrationDocument([state.component], drafts).components[0] || null;
  const metadata = svgNode('metadata', {}, JSON.stringify({ component: state.component.id, mesh_sha256: state.component.mesh_sha256, meters_per_unit: draft().scale, calibration, note: 'Geometric extents include outliers. Room dimensions unverified.' }));
  exported.prepend(metadata);
  const blob = new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n', new XMLSerializer().serializeToString(exported)], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${state.component.id}-denah${draft().scale ? '-kalibrasi-draft' : ''}.svg`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportCalibration() {
  const saved = calibrationDocument(data.components, drafts);
  const blob = new Blob([JSON.stringify(saved, null, 2) + '\n'], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'kalibrasi-draft.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  text('calibration-file-status', saved.components.length ? `JSON diunduh: ${saved.components.length} komponen terkalibrasi draft. Simpan file untuk dimuat kembali pada mesh yang sama.` : 'JSON diunduh tanpa kalibrasi: belum ada komponen dengan acuan ukuran nyata.');
}
async function importCalibration(event) {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const replacements = validateCalibrationDocument(JSON.parse(await file.text()), data.components);
    for (const [id, current] of replacements) drafts.set(id, current);
    $('real-length').value = draft().reference?.meters ?? '';
    updateMeasurement();
    text('calibration-file-status', replacements.size ? `Draft dimuat untuk ${replacements.size} komponen. Komponen lain tetap memakai kalibrasinya sendiri; ukuran belum terverifikasi.` : 'JSON tidak berisi kalibrasi. Draft sebelumnya dipertahankan.');
  } catch (error) {
    text('calibration-file-status', `JSON kalibrasi ditolak. ${error.message} Semua draft sebelumnya dipertahankan.`);
  } finally {
    event.target.value = '';
  }
}

function init() {
  if (data.schema_version !== 1 || !data.components?.length) throw new Error('Format data tur tidak didukung atau komponen kosong');
  text('project-title', data.title || 'Rekonstruksi ruang');
  document.title = `${data.title || 'Rekonstruksi ruang'} / ArchiCon`;
  text('component-count', String(data.components.length).padStart(2, '0'));
  const recordedViews = new Set(data.components.flatMap((component) => component.cameras.map((frame) => frame.frame_id)));
  text('project-summary', `${data.components.length} komponen · ${recordedViews.size} sudut rekaman`);
  text('source', typeof data.source === 'string' ? data.source : JSON.stringify(data.source));
  for (const limitation of data.limitations || []) {
    const item = document.createElement('li'); item.textContent = limitation; $('limitations').append(item);
  }
  for (const component of data.components) {
    const option = document.createElement('option'); option.value = component.id; option.textContent = component.name; $('component').append(option);
    const button = document.createElement('button');
    button.className = 'component-item';
    button.type = 'button';
    button.dataset.component = component.id;
    button.setAttribute('aria-pressed', 'false');
    button.setAttribute('aria-label', `${component.name}, ${component.cameras.length} kamera`);
    const preview = document.createElement('span');
    preview.className = 'component-thumbnail';
    preview.setAttribute('aria-hidden', 'true');
    if (component.floorplan.embedded_image) {
      const image = document.createElement('img');
      image.src = component.floorplan.embedded_image;
      image.alt = '';
      preview.append(image);
    }
    const copy = document.createElement('span');
    copy.className = 'component-copy';
    const title = document.createElement('strong');
    title.textContent = component.name;
    const subtitle = document.createElement('small');
    subtitle.textContent = `${component.cameras.length} kamera rekaman`;
    copy.append(title, subtitle);
    button.append(preview, copy);
    button.addEventListener('click', () => selectComponent(component.id));
    $('component-list').append(button);
  }
  try { init3d(); } catch (error) { fail(new Error(`WebGL tidak tersedia: ${error.message}`)); }
  $('component').addEventListener('change', (event) => selectComponent(event.target.value));
  $('tab-3d').addEventListener('click', () => setTab('3d'));
  $('tab-plan').addEventListener('click', () => setTab('plan'));
  $('open-measure').addEventListener('click', () => setTab('plan'));
  document.querySelector('.view-tabs').addEventListener('keydown', (event) => {
    if (!event.target.matches('[role=tab]') || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const tab = event.key === 'Home' ? '3d' : event.key === 'End' ? 'plan' : state.tab === '3d' ? 'plan' : '3d';
    setTab(tab);
    $(`tab-${tab}`).focus();
  });
  $('overview').addEventListener('click', () => { setTab('3d'); overview(); });
  $('roof').addEventListener('input', updateRoof);
  $('frame').addEventListener('input', (event) => { stop(); showFrame(Number(event.target.value)); });
  $('play').addEventListener('click', play);
  $('details-toggle').addEventListener('click', () => { $('details').hidden = !$('details').hidden; $('details-toggle').setAttribute('aria-expanded', !$('details').hidden); });
  $('reset-points').addEventListener('click', () => { draft().points = []; updateMeasurement(); });
  $('calibration-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const current = draft();
    const meters = Number($('real-length').value);
    if (current.points.length !== 2 || !Number.isFinite(meters) || meters <= 0) return;
    const units = distance(...current.points);
    const scale = meters / units;
    try {
      const candidate = { scale, reference: { meters, units, points: current.points.map((p) => [...p]) } };
      const replacements = validateCalibrationDocument(calibrationDocument([state.component], new Map([[state.component.id, candidate]])), [state.component]);
      drafts.set(state.component.id, replacements.get(state.component.id));
    } catch {
      text('calibration-state', 'Kalibrasi ditolak: pilih segmen yang lebih panjang dan masukkan ukuran nyata yang valid. Skala sebelumnya dipertahankan.');
      return;
    }
    updateMeasurement();
  });
  $('reset-calibration').addEventListener('click', () => { draft().scale = null; draft().reference = null; $('real-length').value = ''; updateMeasurement(); });
  $('export-svg').addEventListener('click', exportSvg);
  $('export-calibration').addEventListener('click', exportCalibration);
  $('load-calibration').addEventListener('click', () => $('calibration-file').click());
  $('calibration-file').addEventListener('change', importCalibration);
  $('evidence-toggle').addEventListener('click', () => {
    state.evidence = !state.evidence;
    $('evidence-toggle').setAttribute('aria-pressed', state.evidence);
    text('evidence-toggle', `Bukti irisan: ${state.evidence ? 'aktif' : 'nonaktif'}`);
    const image = planSvg.querySelector('#plan-evidence');
    if (image) image.style.display = state.evidence ? 'inline' : 'none';
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  selectComponent(data.components[0].id);
  requestAnimationFrame(animate);
}
try { init(); } catch (error) { fail(error); }
