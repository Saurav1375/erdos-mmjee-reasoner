// Records the mmJEE demo website (replay of P2-EN-Math-MCQ-q3) with Chrome's screencast.
// Output: frames/*.jpg + frames.json (timestamps) + moments.json (time and on-screen boxes of every
// labelled moment, in video pixels), then demo.mp4 (30 fps) via ffmpeg.
//   node record.mjs   (site served at http://127.0.0.1:8810)
import { chromium } from 'playwright';
import fs from 'node:fs';
import { execSync } from 'node:child_process';

const QID = 'P2-HI-Chem-Num-q7';
const W = 1440, H = 810, DPR = 4 / 3;            // laid out at 1440x810, captured at 1920x1080
fs.rmSync('frames', { recursive: true, force: true }); fs.mkdirSync('frames');

const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome-stable', headless: true });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: DPR, colorScheme: 'light' });
const page = await ctx.newPage();
await page.goto('http://127.0.0.1:8810/index.html');
await page.waitForTimeout(1200);
await page.evaluate(() => document.fonts.ready);

// Clean capture: no scrollbars, a visible cursor with a click ring.
await page.addStyleTag({ content: `
  ::-webkit-scrollbar { display: none; } * { scrollbar-width: none; }
  html { scroll-behavior: auto !important; }
  #rec-cursor { position: fixed; left: 0; top: 0; width: 22px; height: 22px; z-index: 99999; pointer-events: none;
    transform: translate(-3px, -2px); transition: none; }
  .rec-ring { position: fixed; width: 34px; height: 34px; margin: -17px 0 0 -17px; border-radius: 50%;
    border: 3px solid #ffd20a; pointer-events: none; z-index: 99998; animation: recring .55s ease-out forwards; }
  @keyframes recring { from { transform: scale(.4); opacity: 1; } to { transform: scale(1.6); opacity: 0; } }` });
await page.evaluate(() => {
  const c = document.createElement('div'); c.id = 'rec-cursor';
  c.innerHTML = '<svg width="22" height="22" viewBox="0 0 22 22"><path d="M2 1.5 L2 18 L6.6 13.8 L9.6 20.5 L12.6 19.2 L9.7 12.6 L16 12.6 Z" fill="#17150f" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/></svg>';
  document.body.append(c);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; });
  // Eased scrolling of the window or an element, resolved when finished.
  window.__ease = (el, to, ms) => new Promise(res => {
    const win = el === window, from = win ? scrollY : el.scrollTop, t0 = performance.now();
    const f = t => { const k = Math.min(1, (t - t0) / ms), e = k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      const y = from + (to - from) * e; win ? scrollTo(0, y) : (el.scrollTop = y); k < 1 ? requestAnimationFrame(f) : res(); };
    requestAnimationFrame(f);
  });
});

// ------------------------------------------------------------------ screencast
const cdp = await ctx.newCDPSession(page);
const frames = []; let t0 = null;
cdp.on('Page.screencastFrame', async ({ data, metadata, sessionId }) => {
  const ts = metadata.timestamp; if (t0 === null) t0 = ts;
  const f = `frames/${String(frames.length).padStart(5, '0')}.jpg`;
  fs.writeFileSync(f, Buffer.from(data, 'base64')); frames.push({ f, t: ts - t0 });
  cdp.send('Page.screencastFrameAck', { sessionId }).catch(() => {});
});
const now = () => (t0 === null ? 0 : Date.now() / 1000 - t0);   // screencast timestamps are wall-clock seconds
await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
await page.waitForTimeout(400);

// ------------------------------------------------------------------ helpers
const moments = [];
const box = async sel => page.evaluate(s => {
  const e = document.querySelector(s); if (!e) return null;
  const r = e.getBoundingClientRect(); return [r.x, r.y, r.width, r.height];
}, sel).then(b => b && b.map(v => Math.round(v * DPR)));
async function mark(id, sels = {}) {
  const boxes = {};
  for (const [k, s] of Object.entries(sels)) boxes[k] = await box(s);
  moments.push({ id, t: +now().toFixed(3), boxes });
}
let mx = 900, my = 500;
async function moveTo(sel, dx = 0.5, dy = 0.5, ms = 700) {
  const b = await page.evaluate(s => { const r = document.querySelector(s).getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }, sel);
  const x = b[0] + b[2] * dx, y = b[1] + b[3] * dy, n = Math.max(12, Math.round(ms / 16));
  for (let i = 1; i <= n; i++) {                       // eased path
    const k = i / n, e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    await page.mouse.move(mx + (x - mx) * e, my + (y - my) * e); await page.waitForTimeout(ms / n);
  }
  mx = x; my = y;
}
async function click(sel, dx, dy) {
  await moveTo(sel, dx, dy);
  await page.evaluate(([x, y]) => { const r = document.createElement('div'); r.className = 'rec-ring';
    r.style.left = x + 'px'; r.style.top = y + 'px'; document.body.append(r); setTimeout(() => r.remove(), 700); }, [mx, my]);
  await page.mouse.down(); await page.waitForTimeout(90); await page.mouse.up();
}
const wait = ms => page.waitForTimeout(ms);
const scrollWin = (y, ms = 900) => page.evaluate(([y, ms]) => __ease(window, y, ms), [y, ms]);
const scrollEl = (sel, y, ms = 900) => page.evaluate(([s, y, ms]) => __ease(document.querySelector(s), y, ms), [sel, y, ms]);
const topOf = sel => page.evaluate(s => document.querySelector(s).getBoundingClientRect().top + scrollY, sel);
const stageTop = async () => (await topOf('#replay .stage-wrap')) - 66;   // just under the sticky header
// never scroll past the bottom of the replay panel, so the next section stays out of frame
const clampY = async y => Math.min(y, await page.evaluate(() => {
  const r = document.querySelector('#replay .stage-wrap').getBoundingClientRect(); return r.bottom + scrollY - innerHeight + 24; }));
const step = n => `#rail .step:nth-of-type(${n})`;

// ------------------------------------------------------------------ the walk-through
await mark('hero', { title: '.hero h1', specimen: '#specimen' });
await wait(3000);
await scrollWin(await topOf('#replay') - 40, 1400);
await mark('replay', { head: '#replay .sec-head', picker: '#qpick' });
await wait(1600);

// choose the question
await moveTo('#qpick', 0.5, 0.5, 900);
await page.evaluate(([id]) => { const s = document.querySelector('#qpick'); s.value = id; s.dispatchEvent(new Event('change', { bubbles: true })); }, [QID]);
await wait(300);
await mark('pick', { picker: '#qpick', title: '#qtitle', chips: '#qchips' });
await wait(2200);
await scrollWin(await stageTop(), 1100);

// Stage 1: read the question
await mark('s1', { rail: '#rail', image: '#stage .qimg', transcript: '#stage .tx' });
await wait(3500);
await moveTo('#stage .tx', 0.5, 0.4, 700);
await scrollEl('#stage .tx', 400, 3200);
await wait(900);

// Stage 2: look up solved problems
await click(step(2));
await wait(500);
await mark('s2', { rail: step(2), tag: '#stage .tag', meter: '#stage .meter', examples: '#stage .two > div:last-child' });
await wait(3800);
await moveTo('#stage .meter', 0.62, 0.35, 700);
await mark('s2_tau', { meter: '#stage .meter' });
await wait(3200);
await scrollWin(await clampY(await topOf('#stage .kbcard') - 140), 1200);
await mark('s2_examples', { first: '#stage .kbcard', examples: '#stage .two > div:last-child' });
await wait(4200);
await scrollWin(await stageTop(), 900);

// Stage 3: solve with Python. The stage animates its code run when it opens, so the view is set
// on the cell first and the stage is opened again with the same scroll, to show the run in frame.
await click(step(3));
await wait(300);
await mark('s3', { rail: step(3), stats: '#stage .statrow', sol: '#stage .sol' });
await wait(2600);
const cellAt = () => page.evaluate(() => { const s = document.querySelector('#stage .sol'), c = s.querySelector('.cell');
  return c ? Math.max(0, c.getBoundingClientRect().top - s.getBoundingClientRect().top + s.scrollTop - 150) : 0; });
await scrollWin(await clampY(await topOf('#stage .sol') - 150), 800);
await scrollEl('#stage .sol', await cellAt(), 1600);
await wait(300);
await page.evaluate(() => { const s = document.querySelector('#stage .sol'), y = s.scrollTop;
  __demo.go(2); const n = document.querySelector('#stage .sol'); n.scrollTop = y; });
await mark('s3_code', { cell: '#stage .sol .cell' });
await wait(5200);
await scrollEl('#stage .sol', 1e6, 2400);
await scrollWin(await clampY(await topOf('#stage .answer-line') - 300), 800);
await mark('s3_answer', { answer: '#stage .answer-line', stats: '#stage .statrow' });
await wait(3400);
await scrollWin(await stageTop(), 900);

// Stage 4: critique and correct
await click(step(4));
await wait(400);
await mark('s4', { rail: step(4), r1: '#stage .round', say: '#stage .say' });
await wait(4200);
if (await page.$('#stage .gate')) {
  await scrollWin(await clampY(await topOf('#stage .gate') - 380), 1000);
  await mark('s4_gate', { r1: '#stage .round', gate: '#stage .gate' });
  await wait(3600);
}
await scrollWin(await stageTop(), 900);

// Stage 5: verify and pick
await click(step(5));
await wait(300);
await mark('s5', { rail: step(5), pool: '#stage .pool > div:first-child', pick: '#stage .prow.pick', votes: '#stage .votes' });
await wait(4200);
await moveTo('#stage .prow.pick', 0.55, 0.5, 700);
await mark('s5_pick', { pick: '#stage .prow.pick', callout: '#stage .callout' });
await wait(3800);

// Stage 6: result
await click(step(6));
await wait(400);
await mark('s6', { rail: step(6), table: '#stage .cmp', ours: '#stage .cmp tr.ours', verdict: '#stage .verdict-box' });
await wait(5200);
await mark('end');
await wait(600);

await cdp.send('Page.stopScreencast');
await wait(300);
await browser.close();

// ------------------------------------------------------------------ assemble a constant 30 fps video
fs.writeFileSync('moments.json', JSON.stringify({ fps: 30, width: 1920, height: 1080, question: QID, moments }, null, 1));
fs.writeFileSync('frames.json', JSON.stringify(frames));
const total = moments.at(-1).t + 0.6;
let list = '';
frames.forEach((fr, i) => {
  const next = i + 1 < frames.length ? frames[i + 1].t : total;
  list += `file '${fr.f}'\nduration ${Math.max(0.001, next - fr.t).toFixed(4)}\n`;
});
list += `file '${frames.at(-1).f}'\n`;
fs.writeFileSync('frames.txt', list);
execSync(`ffmpeg -y -loglevel error -f concat -safe 0 -i frames.txt -vf "fps=30,scale=1920:1080:flags=lanczos,format=yuv420p" -t ${total.toFixed(3)} -c:v libx264 -crf 16 -preset slow -movflags +faststart demo.mp4`);
console.log(`frames ${frames.length}, length ${total.toFixed(1)} s, moments ${moments.length}`);
