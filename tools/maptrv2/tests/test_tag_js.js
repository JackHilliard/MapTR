// Exercise TAG_JS in a real DOM against the page HTML the server emitted,
// with fetch stubbed. The python tests cover the /tag endpoint; this covers
// the half that only ever runs in a browser -- which no server-side test and
// no curl can reach.
const fs = require('fs');
const { JSDOM } = require('jsdom');

const html = fs.readFileSync('/w/page.html', 'utf8');
const calls = [];
let nextReply = null;

const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true });
const w = dom.window;
w.fetch = function (url, opts) {
  const body = JSON.parse(opts.body);
  calls.push({ url, body });
  const reply = nextReply || { ok: true, tags: [body.tag], vocab: ['corrupted', body.tag] };
  return Promise.resolve({ json: () => Promise.resolve(reply) });
};

let fails = [];
function ck(name, cond, extra) {
  console.log((cond ? 'PASS ' : 'FAIL ') + name + (extra !== undefined ? '  ' + extra : ''));
  if (!cond) fails.push(name);
}
const tick = () => new Promise(r => setTimeout(r, 0));

(async () => {
  const boxes = w.document.querySelectorAll('input.tagbox');
  ck('checkboxes present in the served HTML', boxes.length > 0, boxes.length);

  // --- ticking a box posts
  const box = boxes[0];
  box.checked = true;
  box.dispatchEvent(new w.Event('change', { bubbles: true }));
  await tick();
  ck('tick -> one POST to /tag', calls.length === 1 && calls[0].url === '/tag',
     JSON.stringify(calls[0]));
  ck('POST carries uid, tag and on=1',
     calls[0].body.uid === box.dataset.uid &&
     calls[0].body.tag === box.dataset.tag && calls[0].body.on === 1,
     JSON.stringify(calls[0].body));

  // --- unticking posts on=0
  box.checked = false;
  box.dispatchEvent(new w.Event('change', { bubbles: true }));
  await tick();
  ck('untick -> on=0', calls[1].body.on === 0, JSON.stringify(calls[1].body));

  // --- a server error reverts the checkbox and shows the message
  nextReply = { ok: false, error: 'cannot write /ro/tile_tags.json' };
  box.checked = true;
  box.dispatchEvent(new w.Event('change', { bubbles: true }));
  await tick(); await tick();
  ck('failed write REVERTS the checkbox', box.checked === false, box.checked);
  const msg = box.closest('.tags').querySelector('.tagmsg');
  ck('failed write shows the error', /cannot write/.test(msg.textContent),
     msg.textContent);
  ck('error is styled as an error', msg.className.indexOf('err') >= 0,
     msg.className);
  nextReply = null;

  // a later success clears the message
  box.checked = true;
  box.dispatchEvent(new w.Event('change', { bubbles: true }));
  await tick(); await tick();
  ck('a later success clears the message', msg.textContent === '', msg.textContent);

  // --- creating a new tag
  const nRows = w.document.querySelectorAll('.tags').length;
  const before = w.document.querySelectorAll('input.tagbox').length;
  const inp = w.document.querySelector('input.tagnew');
  const uid = inp.dataset.uid;
  inp.value = '  occluded  ';
  const ev = new w.KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
  inp.dispatchEvent(ev);
  await tick(); await tick();
  ck('Enter posts the trimmed tag',
     calls[calls.length - 1].body.tag === 'occluded', calls[calls.length - 1].body.tag);
  ck('Enter does not submit the form (prevented)', ev.defaultPrevented);
  const after = w.document.querySelectorAll('input.tagbox').length;
  ck('the new tag gets a box on EVERY tile', after === before + nRows,
     `${before} -> ${after} over ${nRows} rows`);
  const created = w.document.querySelectorAll('input.tagbox[data-tag="occluded"]');
  ck('created tag is ticked only on its own tile',
     Array.from(created).filter(b => b.checked).length === 1 &&
     Array.from(created).find(b => b.checked).dataset.uid === uid);
  ck('the input is cleared', inp.value === '');

  // --- empty / whitespace tag does nothing
  const n = calls.length;
  inp.value = '   ';
  inp.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  await tick();
  ck('an empty new-tag does not post', calls.length === n, calls.length - n);

  // --- a box created by JS is live (delegation, not per-node handlers)
  const fresh = Array.from(w.document.querySelectorAll('input.tagbox'))
    .find(b => b.dataset.tag === 'occluded' && !b.checked);
  const m = calls.length;
  fresh.checked = true;
  fresh.dispatchEvent(new w.Event('change', { bubbles: true }));
  await tick();
  ck('a JS-created checkbox posts too (event delegation)',
     calls.length === m + 1 && calls[calls.length - 1].body.tag === 'occluded');

  console.log('');
  console.log(fails.length ? 'FAILURES:\n  ' + fails.join('\n  ') : 'ALL PASS');
  process.exit(fails.length ? 1 : 0);
})();
