const positive = (value) => Number.isFinite(value) && value > 0;
const close = (a, b) => Math.abs(a - b) <= Math.max(Math.abs(a), Math.abs(b)) * 1e-8;
function check(condition, message) { if (!condition) throw new Error(message); }
function copyReference(reference) {
  return { points: reference.points.map((point) => [...point]), meters: reference.meters, units: reference.units };
}

export function calibrationDocument(components, drafts) {
  return {
    schema_version: 1,
    components: components.filter((component) => drafts.get(component.id)?.scale != null).map((component) => {
      const current = drafts.get(component.id);
      return {
        component_id: component.id,
        mesh_sha256: component.mesh_sha256,
        meters_per_unit: current.scale,
        reference: copyReference(current.reference),
        measurement_source: 'user_entered',
        validation_status: 'draft',
      };
    }),
  };
}

// Return a complete set of replacements before the caller changes any draft.
// Components absent from the document retain their own calibration.
export function validateCalibrationDocument(document, components) {
  check(document?.schema_version === 1 && Array.isArray(document.components), 'Format JSON kalibrasi tidak didukung.');
  const available = new Map(components.map((component) => [component.id, component]));
  const replacements = new Map();
  for (const record of document.components) {
    const component = available.get(record?.component_id);
    check(component, 'Komponen kalibrasi tidak ditemukan pada HTML ini.');
    check(!replacements.has(component.id), `Kalibrasi komponen duplikat: ${component.id}.`);
    check(typeof component.mesh_sha256 === 'string' && /^[a-f0-9]{64}$/.test(component.mesh_sha256) && record.mesh_sha256 === component.mesh_sha256, `Mesh SHA-256 tidak cocok: ${component.id}.`);
    check(record.measurement_source === 'user_entered' && record.validation_status === 'draft', 'Sumber pengukuran harus user_entered dan status harus draft.');
    const { meters_per_unit: scale, reference } = record;
    check(positive(scale) && positive(reference?.meters) && positive(reference?.units), 'Skala, meter, dan satuan acuan harus angka positif hingga.');
    check(Array.isArray(reference.points) && reference.points.length === 2 && reference.points.every((point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite)), 'Dua titik acuan X/Z harus valid.');
    const [a, b] = reference.points;
    const units = Math.hypot(a[0] - b[0], a[1] - b[1]);
    check(Number.isFinite(units) && units >= 1e-8 && close(units, reference.units), 'Jarak titik tidak cocok dengan satuan acuan.');
    check(positive(reference.meters / reference.units) && close(scale, reference.meters / reference.units), 'Skala tidak cocok dengan meter / satuan acuan.');
    for (const bounds of [component.bounds, component.floorplan.bounds]) {
      const spans = bounds[0].map((min, index) => bounds[1][index] - min);
      check(spans.every((span) => Number.isFinite(span) && span >= 0 && Number.isFinite(span * scale)) && Number.isFinite(Math.hypot(...spans) * scale), 'Kalibrasi melampaui rentang bentang geometri.');
    }
    replacements.set(component.id, { scale, reference: copyReference(reference), points: reference.points.map((point) => [...point]) });
  }
  return replacements;
}
