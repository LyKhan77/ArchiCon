#!/usr/bin/env node
import { readFile, writeFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';

const directory = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const options = {};
for (let i = 0; i < args.length; i += 2) {
  if (!['--data', '--output'].includes(args[i]) || !args[i + 1]) {
    console.error('Usage: node web/build.mjs --data PATH/tour-data.json --output PATH/tour.html');
    process.exit(1);
  }
  options[args[i].slice(2)] = path.resolve(args[i + 1]);
}
if (!options.data || !options.output) {
  console.error('Usage: node web/build.mjs --data PATH/tour-data.json --output PATH/tour.html');
  process.exit(1);
}

function check(condition, message) { if (!condition) throw new Error(message); }
function validateGLB(buffer, name) {
  check(buffer.length >= 20 && buffer.toString('ascii', 0, 4) === 'glTF', `${name}: not a GLB`);
  check(buffer.readUInt32LE(4) === 2 && buffer.readUInt32LE(8) === buffer.length, `${name}: invalid GLB header`);
  const jsonLength = buffer.readUInt32LE(12);
  check(buffer.readUInt32LE(16) === 0x4e4f534a && 20 + jsonLength <= buffer.length, `${name}: invalid GLB JSON chunk`);
  const gltf = JSON.parse(buffer.toString('utf8', 20, 20 + jsonLength));
  for (const item of [...(gltf.buffers || []), ...(gltf.images || [])]) {
    check(!item.uri || item.uri.startsWith('data:'), `${name}: external asset URI forbidden in offline viewer`);
  }
  const supported = new Set(['KHR_mesh_quantization', 'KHR_materials_unlit', 'KHR_materials_pbrSpecularGlossiness', 'KHR_texture_transform', 'EXT_mesh_gpu_instancing']);
  for (const extension of gltf.extensionsRequired || []) check(supported.has(extension), `${name}: required extension ${extension} not supported by this offline build`);
}
function safeScript(value) { return value.replace(/<\/script/gi, '<\\/script').replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029'); }

try {
  // Exclusive output is also enforced on write, so a concurrent build cannot overwrite it.
  try { await stat(options.output); throw new Error(`Output already exists: ${options.output}`); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const data = JSON.parse(await readFile(options.data, 'utf8'));
  check(data.schema_version === 1 && data.components?.length, 'Expected schema_version 1 and at least one component');
  const ids = new Set();
  for (const component of data.components) {
    check(typeof component.id === 'string' && component.id && !ids.has(component.id), 'Component IDs must be nonempty and unique');
    ids.add(component.id);
    check(component.bounds?.length === 2 && component.bounds.every((p) => p.length === 3 && p.every(Number.isFinite)), `${component.id}: invalid bounds`);
    check(component.floorplan?.bounds?.length === 2 && Array.isArray(component.floorplan.segments) && Array.isArray(component.floorplan.path), `${component.id}: invalid floorplan`);
    check(Array.isArray(component.cameras), `${component.id}: cameras must be an array`);
    let previous = -Infinity;
    for (const camera of component.cameras) {
      check(Number.isFinite(camera.time) && camera.time >= previous, `${component.id}: camera times must be finite and ordered`);
      check(camera.matrix?.length === 4 && camera.matrix.every((row) => row.length === 4 && row.every(Number.isFinite)), `${component.id}: invalid row-major camera matrix`);
      check(camera.position?.length === 3 && camera.position.every(Number.isFinite), `${component.id}: invalid camera position`);
      previous = camera.time;
    }
    const mesh = await readFile(path.resolve(path.dirname(options.data), component.mesh_file));
    validateGLB(mesh, component.id);
    component.mesh_sha256 = createHash('sha256').update(mesh).digest('hex');
    component.embedded_glb = mesh.toString('base64');
    if (component.floorplan.image_file) {
      const imagePath = path.resolve(path.dirname(options.data), component.floorplan.image_file);
      const image = await readFile(imagePath);
      const mime = imagePath.toLowerCase().endsWith('.svg') ? 'image/svg+xml' : 'image/png';
      component.floorplan.embedded_image = `data:${mime};base64,${image.toString('base64')}`;
    }
  }
  const bundle = await build({ entryPoints: [path.join(directory, 'tour/main.js')], bundle: true, format: 'iife', platform: 'browser', target: ['es2020'], minify: true, legalComments: 'inline', write: false });
  const [template, css, threeLicense, fontRegular, fontLight, fontLicense] = await Promise.all([
    readFile(path.join(directory, 'tour/index.html'), 'utf8'),
    readFile(path.join(directory, 'tour/style.css'), 'utf8'),
    readFile(path.join(directory, 'node_modules/three/LICENSE'), 'utf8'),
    readFile(path.join(directory, 'dashboard/fonts/IBMPlexSans-Regular.woff2')),
    readFile(path.join(directory, 'dashboard/fonts/IBMPlexSans-Light.woff2')),
    readFile(path.join(directory, 'dashboard/fonts/OFL.txt'), 'utf8'),
  ]);
  const offlineCss = css.replace('/*__FONT_REGULAR__*/', () => fontRegular.toString('base64'))
    .replace('/*__FONT_LIGHT__*/', () => fontLight.toString('base64'))
    + `\n/* IBM Plex Sans license\n${fontLicense.replace(/\*\//g, '* /')}\n*/`;
  const html = template.replace('/*__STYLE__*/', () => offlineCss)
    .replace('/*__DATA__*/', () => safeScript(JSON.stringify(data).replace(/</g, '\\u003c')))
    .replace('/*__BUNDLE__*/', () => safeScript(`/* Three.js license\n${threeLicense.replace(/\*\//g, '* /')}\n*/\n${bundle.outputFiles[0].text}`));
  await writeFile(options.output, html, { flag: 'wx' });
  console.log(`Built ${options.output} (${(Buffer.byteLength(html) / 1024 / 1024).toFixed(2)} MiB), ${data.components.length} component(s).`);
} catch (error) { console.error(`Build failed: ${error.message}`); process.exitCode = 1; }
