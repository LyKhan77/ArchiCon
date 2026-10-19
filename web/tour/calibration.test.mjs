import test from 'node:test';
import assert from 'node:assert/strict';
import { calibrationDocument, validateCalibrationDocument } from './calibration.js';

const components = ['a', 'b'].map((id, index) => ({ id, mesh_sha256: String(index + 1).repeat(64), bounds: [[0, 0, 0], [10, 4, 8]], floorplan: { bounds: [[0, 0], [10, 8]] } }));
const record = (id = 'a') => ({ component_id: id, mesh_sha256: components.find((c) => c.id === id).mesh_sha256, meters_per_unit: 2, reference: { points: [[0, 0], [3, 4]], meters: 10, units: 5 }, measurement_source: 'user_entered', validation_status: 'draft' });
const document = (...records) => ({ schema_version: 1, components: records });

test('save and restore independent component references with no inferred scales', () => {
  const drafts = new Map([['a', { scale: 2, reference: record().reference, points: [[1, 1]] }], ['b', { scale: null, reference: null, points: [] }]]);
  const exported = calibrationDocument(components, drafts);
  assert.deepEqual(exported, document(record()));
  const loaded = validateCalibrationDocument(JSON.parse(JSON.stringify(exported)), components);
  assert.equal(loaded.size, 1);
  assert.equal(loaded.get('a').scale, 2);
  assert.deepEqual(loaded.get('a').points, [[0, 0], [3, 4]]);
  assert.equal(loaded.has('b'), false);
  loaded.get('a').reference.points[0][0] = 9;
  assert.equal(drafts.get('a').reference.points[0][0], 0);
});

test('rejects unknown components, duplicate records and mismatched exact mesh hashes', () => {
  const badHash = { ...record(), mesh_sha256: 'f'.repeat(64) };
  assert.throws(() => validateCalibrationDocument(document(badHash), components), /mesh/i);
  assert.throws(() => validateCalibrationDocument(document({ ...record(), component_id: 'unknown' }), components), /komponen|component/i);
  assert.throws(() => validateCalibrationDocument(document(record(), record()), components), /duplikat|duplicate/i);
});

test('rejects corrupt documents, false provenance and invalid physical values', () => {
  for (const value of [null, {}, { schema_version: 2, components: [] }, document(null)]) assert.throws(() => validateCalibrationDocument(value, components));
  for (const scale of [0, -1, Infinity, NaN, '2', 1e308]) assert.throws(() => validateCalibrationDocument(document({ ...record(), meters_per_unit: scale }), components));
  for (const patch of [{ measurement_source: 'estimated' }, { validation_status: 'verified' }, { reference: { ...record().reference, meters: 0 } }, { reference: { ...record().reference, units: 0 } }, { reference: { ...record().reference, points: [[0, 0], [Infinity, 0]] } }]) assert.throws(() => validateCalibrationDocument(document({ ...record(), ...patch }), components));
});

test('requires reference distance and scale to agree with entered values', () => {
  assert.throws(() => validateCalibrationDocument(document({ ...record(), reference: { ...record().reference, units: 6 } }), components));
  assert.throws(() => validateCalibrationDocument(document({ ...record(), meters_per_unit: 3 }), components));
  assert.throws(() => validateCalibrationDocument(document({ ...record(), reference: { ...record().reference, points: [[0, 0], [0, 0]] } }), components));
});

test('validates every record before producing replacements; input stays unchanged', () => {
  const source = document(record(), { ...record('b'), mesh_sha256: 'f'.repeat(64) });
  const before = JSON.stringify(source);
  assert.throws(() => validateCalibrationDocument(source, components));
  assert.equal(JSON.stringify(source), before);
});

test('empty export and empty import contain no calibration assumptions', () => {
  assert.deepEqual(calibrationDocument(components, new Map()), document());
  assert.equal(validateCalibrationDocument(document(), components).size, 0);
});

test('rejects scaled mesh extents that overflow even with a consistent reference', () => {
  const huge = { ...components[0], bounds: [[-1e308, 0, 0], [1e308, 1, 1]] };
  assert.throws(() => validateCalibrationDocument(document(record()), [huge]));
});
