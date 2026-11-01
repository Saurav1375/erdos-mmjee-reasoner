// Records a LIVE run of the frozen system on the lab GPUs through the demo page (localhost:8765,
// started with demo_website/live/start.sh). Captures ~30 fps while something worth watching at
// normal speed happens and ~4 fps during the waits (attempts being written, critic checking),
// which the edit fast-forwards. Writes live_frames/, live_events.json (every server event the
// page received, with wall-clock time), live_moments.json and live.mp4.
//   node live.mjs
import { chromium } from 'playwright';
import fs from 'node:fs';
import { execSync } from 'node:child_process';

const QID = 'P1-EN-Phys-MCQM-q3';
const W = 1440, H = 810, DPR = 4 / 3;
fs.rmSync('live_frames', { recursive: true, force: true }); fs.mkdirSync('live_frames');

const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome-stable', headless: true });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: DPR, colorScheme: 'light' });
// Log every server event the page gets (except token deltas, which are only counted).
await ctx.addInitScript(() => {
  const ES = window.EventSource;
  window.__ev = []; window.__deltas = 0;
  window.EventSource = function (url, opts) {
    const es = new ES(url, opts);
    es.addEventListener('message', m => {
      try {
        const d = JSON.parse(m.data);
        if (d.type === 'delta') { window.__deltas++; return; }
        window.__ev.push({ t: Date.now() / 1000, type: d.type, i: d.i, status: d.status, role: d.role, n: d.n,
          round: d.round, rid: d.rid, kinds: d.kinds, seconds: d.seconds,
          retrieval: d.retrieval ? d.retrieval.map(r => [r.id, r.sim]) : undefined,
          chains: d.chains ? d.chains.map(c => ({ s: c.s, error: c.error, step: c.step, answer: c.answer })) : undefined,
          final: d.record ? d.record.methods.final : undefined });
      } catch { /* ignore */ }
    });
    return es;
  };
  window.EventSource.prototype = ES.prototype;
});
const page = await ctx.newPage();
await page.goto('http://127.0.0.1:8765/');
await page.waitForSelector('#live:not([hidden])', { timeout: 20000 });
await page.waitForTimeout(1500);
await page.evaluate(() => document.fonts.ready);

await page.addStyleTag({ content: `
  ::-webkit-scrollbar { display: none; } * { scrollbar-width: none; }
  html { scroll-behavior: auto !important; }
  #rec-cursor { position: fixed; left: 0; top: 0; width: 22px; height: 22px; z-index: 99999; pointer-events: none; transform: translate(-3px, -2px); }
  .rec-ring { position: fixed; width: 34px; height: 34px; margin: -17px 0 0 -17px; border-radius: 50%;
    border: 3px solid #ffd20a; pointer-events: none; z-index: 99998; animation: recring .55s ease-out forwards; }
  @keyframes recring { from { transform: scale(.4); opacity: 1; } to { transform: scale(1.6); opacity: 0; } }` });
await page.evaluate(() => {
  const c = document.createElement('div'); c.id = 'rec-cursor';
  c.innerHTML = '<svg width="22" height="22" viewBox="0 0 22 22"><path d="M2 1.5 L2 18 L6.6 13.8 L9.6 20.5 L12.6 19.2 L9.7 12.6 L16 12.6 Z" fill="#17150f" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/></svg>';
  document.body.append(c);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; });
  window.__ease = (el, to, ms) => new Promise(res => {
    const win = el === window, from = win ? scrollY : el.scrollTop, t0 = performance.now();
    const f = t => { const k = Math.min(1, (t - t0) / ms), e = k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      const y = from + (to - from) * e; win ? scrollTo(0, y) : (el.scrollTop = y); k < 1 ? requestAnimationFrame(f) : res(); };
    requestAnimationFrame(f);
  });
});

// ------------------------------------------------------------------ screencast with a variable rate
const cdp = await ctx.newCDPSession(page);
const frames = []; let t0 = null; let gapMs = 30;      // ack delay: ~30 fps normally, ~4 fps while waiting
cdp.on('Page.screencastFrame', async ({ data, metadata, sessionId }) => {
  if (t0 === null) t0 = metadata.timestamp;
  const f = `live_frames/${String(frames.length).padStart(6, '0')}.jpg`;
  fs.writeFileSync(f, Buffer.from(data, 'base64')); frames.push({ f, t: metadata.timestamp - t0 });
  setTimeout(() => cdp.send('Page.screencastFrameAck', { sessionId }).catch(() => {}), gapMs);
});
const now = () => Date.now() / 1000 - t0;
await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 90, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
await page.waitForTimeout(400);

// ------------------------------------------------------------------ helpers (as in record.mjs)
const moments = [];
const boxOf = sel => page.evaluate(s => { const e = document.querySelector(s); if (!e || e.closest('[hidden]')) return null;
  const r = e.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }, sel).then(b => b && b.map(v => Math.round(v * DPR)));
async function mark(id, sels = {}) {
  const boxes = {}; for (const [k, s] of Object.entries(sels)) boxes[k] = await boxOf(s);
  moments.push({ id, t: +now().toFixed(3), boxes }); console.log(`${now().toFixed(1)}s  ${id}`);
}
let mx = 900, my = 500;
async function moveTo(sel, dx = 0.5, dy = 0.5, ms = 700) {
  const b = await page.evaluate(s => { const r = document.querySelector(s).getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }, sel);
  const x = b[0] + b[2] * dx, y = b[1] + b[3] * dy, n = Math.max(12, Math.round(ms / 16));
  for (let i = 1; i <= n; i++) { const k = i / n, e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    await page.mouse.move(mx + (x - mx) * e, my + (y - my) * e); await page.waitForTimeout(ms / n); }
  mx = x; my = y;
}
async function click(sel, dx, dy) {
  await moveTo(sel, dx, dy);
  await page.evaluate(([x, y]) => { const r = document.createElement('div'); r.className = 'rec-ring';
    r.style.left = x + 'px'; r.style.top = y + 'px'; document.body.append(r); setTimeout(() => r.remove(), 700); }, [mx, my]);
  await page.mouse.down(); await page.waitForTimeout(90); await page.mouse.up();
}
const wait = ms => page.waitForTimeout(ms);
const scrollWin = (y, ms = 900) => page.evaluate(([y, ms]) => __ease(window, Math.max(0, y), ms), [y, ms]);
const topOf = sel => page.evaluate(s => document.querySelector(s).getBoundingClientRect().top + scrollY, sel);
const visible = sel => page.evaluate(s => { const e = document.querySelector(s); return !!e && !e.hidden && e.getBoundingClientRect().height > 0; }, sel);
const events = () => page.evaluate(() => window.__ev);
const has = async pred => (await events()).some(pred);

// ------------------------------------------------------------------ 1. the form
await scrollWin(await topOf('#live') - 30, 1200);
await mark('intro', { head: '#live .sec-head', form: '#liveForm' });
await wait(2500);
await moveTo('#liveQ', 0.5, 0.5, 800);
await page.evaluate(([id]) => { const s = document.querySelector('#liveQ'); s.value = id; s.dispatchEvent(new Event('change', { bubbles: true })); }, [QID]);
await wait(400);
await mark('choose', { select: '#liveQ', modes: '#liveForm .modes' });
await wait(2600);
await click('#liveGo');
await mark('go', { out: '#liveOut' });

// ------------------------------------------------------------------ 2. follow the run
// Keep the bottom of the live panel at the bottom of the screen (newest output in view), hold
// on the transcript and the retrieval result, and slow the capture down while the run only waits.
const PANELS = ['#lvRead', '#lvLook', '#lvSolve', '#lvCheck', '#lvDone'];
const SELS = { '#lvRead': { panel: '#lvRead' }, '#lvLook': { panel: '#lvLook' },
  '#lvSolve': { panel: '#lvSolve', chips: '#lvSamples', stream: '#lvFeat' }, '#lvCheck': { panel: '#lvCheck' }, '#lvDone': { panel: '#lvDone' } };
const follow = async (ms = 500) => {
  const y = await page.evaluate(() => { const o = document.querySelector('#liveOut').getBoundingClientRect();
    return Math.max(o.top + scrollY - 70, o.bottom + scrollY - innerHeight + 28); });
  const cur = await page.evaluate(() => scrollY);
  if (Math.abs(y - cur) > 90) await scrollWin(y, ms);
};
let focus = null, phase = 'normal', solveSeen = 0, criticSeen = 0, holdUntil = 0;
const setPhase = async (p, id) => { if (phase === p) return; phase = p; gapMs = p === 'wait' ? 250 : 30; await mark(id); };
const deadline = Date.now() + 25 * 60 * 1000;
while (Date.now() < deadline) {
  const ev = await events();
  if (ev.some(e => ['done', 'error', 'stopped'].includes(e.type))) break;
  let latest = null;
  for (const p of PANELS) if (await visible(p)) latest = p;
  if (latest && latest !== focus) {
    focus = latest;
    await follow(800);
    await mark('panel' + latest.replace('#lv', '_').toLowerCase(), SELS[latest]);
    if (latest === '#lvLook') { await wait(6500); holdUntil = Date.now() / 1000 + 12; }   // read the retrieval result
    continue;
  }
  await follow(450);
  const solveStart = ev.find(e => e.type === 'stage' && e.i === 2 && e.status === 'start');
  if (solveStart && !solveSeen && Date.now() / 1000 - solveStart.t > 20 && Date.now() / 1000 > holdUntil) {
    solveSeen = 1; await setPhase('wait', 'ff_solve');
  }
  const solveDone = ev.find(e => e.type === 'stage' && e.i === 2 && e.status === 'done');
  if (solveDone && solveSeen === 1) { solveSeen = 2; await setPhase('normal', 'solve_done'); await follow(600); await wait(3500); }
  const critStart = ev.find(e => e.type === 'stage' && e.i === 3 && e.status === 'start');
  if (critStart && solveSeen === 2 && !criticSeen && Date.now() / 1000 - critStart.t > 4) { criticSeen = 1; await setPhase('wait', 'ff_critic'); }
  const verifyStart = ev.find(e => e.type === 'stage' && e.i === 4 && e.status === 'start');
  if (verifyStart && phase === 'wait') await setPhase('normal', 'verify');
  await wait(250);
}
await setPhase('normal', 'finishing');
await wait(800);

// ------------------------------------------------------------------ 3. the result, then step through it
await follow(900);
await mark('done', { done: '#lvDone', final: '#lvDone .lv-final', check: '#lvCheck', status: '#lvStatus' });
await wait(6500);
await click('#lvOpen');
await wait(1500);
await scrollWin(await topOf('#replay .stage-wrap') - 66, 800);
await mark('replay', { title: '#qtitle' });
await wait(2500);
await click('#rail .step:nth-of-type(2)');
await wait(500);
await mark('replay_look', { examples: '#stage .two > div:last-child', meter: '#stage .meter' });
await wait(4200);
await click('#rail .step:nth-of-type(5)');
await wait(400);
await mark('replay_verify', { pick: '#stage .prow.pick', pool: '#stage .pool > div:first-child' });
await wait(4200);
await click('#rail .step:nth-of-type(6)');
await wait(400);
await mark('replay_result', { table: '#stage .cmp', ours: '#stage .cmp tr.ours' });
await wait(4800);
await mark('end');
await wait(500);

const evlog = (await events()).map(e => ({ ...e, t: +(e.t - t0).toFixed(3) }));
await cdp.send('Page.stopScreencast');
await wait(300);
await browser.close();

// ------------------------------------------------------------------ assemble
fs.writeFileSync('live_events.json', JSON.stringify(evlog, null, 1));
fs.writeFileSync('live_moments.json', JSON.stringify({ fps: 30, width: 1920, height: 1080, question: QID, moments }, null, 1));
const total = moments.at(-1).t + 0.5;
let list = '';
frames.forEach((fr, i) => {
  const next = i + 1 < frames.length ? frames[i + 1].t : total;
  list += `file '${fr.f}'\nduration ${Math.max(0.001, next - fr.t).toFixed(4)}\n`;
});
list += `file '${frames.at(-1).f}'\n`;
fs.writeFileSync('live_frames.txt', list);
execSync(`ffmpeg -y -loglevel error -f concat -safe 0 -i live_frames.txt -vf "fps=30,scale=1920:1080:flags=lanczos,format=yuv420p" -t ${total.toFixed(3)} -c:v libx264 -crf 17 -preset medium -movflags +faststart live.mp4`);
console.log(`frames ${frames.length}, length ${total.toFixed(1)} s, events ${evlog.length}, moments ${moments.length}`);
