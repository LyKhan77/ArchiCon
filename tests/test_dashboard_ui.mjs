import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const root = new URL('../', import.meta.url);
const read = path => readFile(new URL(path, root), 'utf8');

test('dashboard exposes the simple global review session contract', async () => {
  const [html, js, css] = await Promise.all([
    read('web/dashboard/index.html'),
    read('web/dashboard/dashboard.js'),
    read('web/dashboard/dashboard.css'),
  ]);
  assert.match(html, /id="review-session"/);
  assert.match(html, /value="accept"/);
  assert.match(html, /value="needs_review"/);
  assert.match(html, /value="reject"/);
  assert.match(html, /id="review-reason"/);
  assert.match(js, /\/review/);
  assert.match(js, /accepted_partial/);
  assert.match(css, /review-session/);
});
