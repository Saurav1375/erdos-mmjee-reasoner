import React from 'react';
import {
  AbsoluteFill, Img, OffthreadVideo, Sequence, staticFile, interpolate, spring,
  useCurrentFrame, useVideoConfig, Easing,
} from 'remotion';
import { loadFont as loadNarrow } from '@remotion/google-fonts/ArchivoNarrow';
import { loadFont as loadUi } from '@remotion/google-fonts/Archivo';
import { loadFont as loadSerif } from '@remotion/google-fonts/SourceSerif4';
import { loadFont as loadMono } from '@remotion/google-fonts/IBMPlexMono';
import rec from './moments.json';

// Same identity as the demo website: answer-sheet paper, black ink, highlighter yellow.
export const C = {
  paper: '#f4f1e6', sheet: '#fffdf5', ink: '#17150f', ink2: '#4a463a', ink3: '#7a7464',
  rule: '#d8d1bd', hi: '#ffd20a', mark: '#ffe04a', accent: '#835800', right: '#17764a', wrong: '#b3261e',
};
export const F = {
  display: loadNarrow('normal', { weights: ['600', '700'] }).fontFamily,
  ui: loadUi('normal', { weights: ['500', '600', '700'] }).fontFamily,
  serif: loadSerif('normal', { weights: ['400', '600'] }).fontFamily,
  mono: loadMono('normal', { weights: ['400', '500', '600'] }).fontFamily,
};

export const FPS = 30;
export const S = (s: number) => Math.round(s * FPS);

// ------------------------------------------------------------------ timeline
const TITLE = S(5), QUESTION = S(6.5), PIPELINE = S(7);
const REC_LEN = S(rec.moments[rec.moments.length - 1].t + 0.5);
const RESULTS = S(8), END = S(5);
const T_TITLE = 0, T_Q = T_TITLE + TITLE, T_PIPE = T_Q + QUESTION, T_REC = T_PIPE + PIPELINE;
const T_RES = T_REC + REC_LEN, T_END = T_RES + RESULTS;
export const TOTAL = T_END + END;

// ------------------------------------------------------------------ what each recorded moment says
type Box = [number, number, number, number];
type Hl = { key: string; label: string; dur?: number };
type Beat = { stage: number; caption: string; zoom?: number; focus?: string; hl?: Hl[] };
export const STAGES = ['Read the question', 'Look up solved problems', 'Solve with Python', 'Critique and correct', 'Verify and pick', 'Result'];
export const SHORT = ['Read', 'Look up', 'Solve', 'Critique', 'Verify', 'Result'];
export const WHO = ['Qwen3-VL-8B', 'bge-small + FAISS', 'Qwen3-VL-8B + sandbox', 'InternVL3.5-8B', 'XGBoost ranker', "benchmark's scorer"];
const BEATS: Record<string, Beat> = {
  hero: { stage: 0, caption: 'The project website replays saved runs of the frozen system on every JEE Advanced 2026 question.',
    hl: [{ key: 'specimen', label: 'Real 2026 question and its saved answers', dur: 2.9 }] },
  replay: { stage: 0, caption: 'The Replay section follows a single question through all six stages of the pipeline.',
    hl: [{ key: 'picker', label: 'Any of the 204 questions', dur: 2.4 }] },
  pick: { stage: 0, caption: 'We open Paper 2, Chemistry, Numerical Q7. The question is in Hindi.',
    hl: [{ key: 'title', label: 'Selected question', dur: 2.1 }] },
  s1: { stage: 1, caption: 'Qwen3-VL-8B reads the Hindi image and writes the question out in English. Every attempt later sees both.',
    hl: [{ key: 'image', label: 'Question image (Hindi)', dur: 8.2 }, { key: 'transcript', label: 'English transcript', dur: 8.2 }] },
  s2: { stage: 2, caption: 'The model tags the question with its topic and method. This tag is the search key for retrieval.',
    hl: [{ key: 'tag', label: 'Concept tag', dur: 3.7 }] },
  s2_tau: { stage: 2, zoom: 1.32, focus: 'meter', caption: 'Knowledge-base problems are ranked by similarity. Only matches above the threshold τ = 0.882 are used.',
    hl: [{ key: 'meter', label: 'Two matches above τ', dur: 3.1 }] },
  s2_examples: { stage: 2, zoom: 1.14, focus: 'first', caption: "Two solved Raoult's-law problems pass (JEEBench 2018, similarity 0.900; JEE 2020, 0.895). They are added to the prompt as worked examples.",
    hl: [{ key: 'first', label: 'Retrieved example · similarity 0.900', dur: 4.1 }] },
  s3: { stage: 3, caption: 'Eight attempts are sampled. Each one writes its calculations as Python instead of doing arithmetic in text.',
    hl: [{ key: 'stats', label: 'Attempt 8 of 8', dur: 2.4 }] },
  s3_code: { stage: 3, zoom: 1.1, focus: 'cell', caption: 'Generation stops at each code block. The block runs in an isolated sandbox (whitelisted imports, 10 s and 1 GB limits) and the model continues.',
    hl: [{ key: 'cell', label: 'Python written by the model, run in the sandbox', dur: 5.0 }] },
  s3_answer: { stage: 3, caption: 'This attempt ends with the answer 0.16.',
    hl: [{ key: 'answer', label: 'Answer of this attempt', dur: 3.3 }] },
  s4: { stage: 4, zoom: 1.12, focus: 'r1', caption: 'InternVL3.5-8B, a model from another family, checks the solution for its first wrong step. It finds none, so the answer stays.',
    hl: [{ key: 'r1', label: 'Critic verdict', dur: 4.1 }] },
  s5: { stage: 5, caption: 'All 16 answers (8 originals and 8 after correction) are scored by the trained XGBoost verifier.',
    hl: [{ key: 'pool', label: '16 candidate answers', dur: 4.1 }] },
  s5_pick: { stage: 5, zoom: 1.25, focus: 'pick', caption: 'The highest-scoring answer becomes the final answer: 0.16. A plain vote agrees, with 15 of the 16 answers.',
    hl: [{ key: 'pick', label: "Verifier's pick", dur: 3.7 }] },
  s6: { stage: 6, zoom: 1.08, focus: 'table', caption: 'Scored with the benchmark’s own code: correct. A single chain-of-thought answer gets this question wrong (right in only 12 of 20 tries).',
    hl: [{ key: 'ours', label: 'Our framework', dur: 5.1 }] },
  end: { stage: 6, caption: '' },
};

// ------------------------------------------------------------------ small pieces
export const ease = Easing.bezier(0.22, 1, 0.36, 1);
export const fadeIO = (f: number, len: number, d = 12) =>
  interpolate(f, [0, d, len - d, len], [0, 1, 1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
export const rise = (f: number, at: number, dist = 24) => {
  const k = interpolate(f, [at, at + 18], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease });
  return { opacity: k, transform: `translateY(${(1 - k) * dist}px)` };
};
export const Eyebrow: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 14, font: `500 22px ${F.mono}`, letterSpacing: '0.08em',
    textTransform: 'uppercase', color: C.ink, ...style }}>
    <span style={{ width: 16, height: 16, background: C.hi, outline: `2px solid ${C.ink}`, outlineOffset: -2 }} />{children}
  </div>
);
export const Mark: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span style={{ background: `linear-gradient(transparent 50%, ${C.mark} 50%, ${C.mark} 92%, transparent 92%)`, padding: '0 4px' }}>{children}</span>
);

// ------------------------------------------------------------------ 1. title
const Title: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: C.paper, opacity: fadeIO(f, TITLE, 14), padding: '0 150px', justifyContent: 'center' }}>
      <div style={rise(f, 4)}><Eyebrow>mmJEE Reasoner · project demo</Eyebrow></div>
      <div style={{ ...rise(f, 12, 30), font: `700 118px/1.0 ${F.display}`, color: C.ink, marginTop: 34, maxWidth: 1450, letterSpacing: '-0.01em' }}>
        One JEE Advanced question, <Mark>from image to answer</Mark>
      </div>
      <div style={{ ...rise(f, 24), font: `400 36px/1.45 ${F.serif}`, color: C.ink2, marginTop: 40, maxWidth: 1300 }}>
        A frozen Qwen3-VL-8B with a code sandbox, a cross-model critic, a learned verifier and retrieval of solved problems, running on one 12&nbsp;GB GPU.
      </div>
      <div style={{ ...rise(f, 34), marginTop: 70, paddingTop: 22, borderTop: `2px solid ${C.ink}`, font: `500 26px ${F.ui}`, color: C.ink2, display: 'flex', gap: 48 }}>
        <span><b style={{ color: C.ink }}>Team Erdős</b></span><span>Arpit Kumar · Saurav Gupta</span><span>IIT Bhilai · Machine Learning course project</span>
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ 2. the question
const Question: React.FC = () => {
  const f = useCurrentFrame();
  const rows: [string, string][] = [['Exam', 'JEE Advanced 2026, Paper 2'], ['Subject', 'Chemistry · Numerical · Q7'], ['Language', 'Hindi (cropped by us from the official paper)'], ['Answer key', '0.16']];
  return (
    <AbsoluteFill style={{ background: C.paper, opacity: fadeIO(f, QUESTION, 14), padding: '0 120px', flexDirection: 'row', alignItems: 'center', gap: 80 }}>
      <div style={{ ...rise(f, 4, 30), width: 960, flexShrink: 0, background: '#fff', border: `2px solid ${C.ink}`, boxShadow: `14px 14px 0 ${C.hi}`, padding: 18 }}>
        <Img src={staticFile('question.webp')} style={{ width: '100%', display: 'block' }} />
      </div>
      <div style={{ flex: 1 }}>
        <div style={rise(f, 8)}><Eyebrow>The question</Eyebrow></div>
        <div style={{ ...rise(f, 14), font: `700 72px/1.02 ${F.display}`, color: C.ink, margin: '26px 0 34px' }}>A vapour-pressure problem, given only as a Hindi image</div>
        {rows.map(([k, v], i) => (
          <div key={k} style={{ ...rise(f, 22 + i * 5), display: 'grid', gridTemplateColumns: '210px 1fr', padding: '14px 0', borderTop: `1px solid ${C.rule}`, font: `400 27px/1.35 ${F.serif}`, color: C.ink }}>
            <span style={{ font: `500 20px ${F.mono}`, letterSpacing: '0.06em', textTransform: 'uppercase', color: C.ink3, paddingTop: 5 }}>{k}</span>
            <span style={k === 'Answer key' ? { font: `600 30px ${F.mono}` } : {}}>{v}</span>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ 3. pipeline overview
const Pipeline: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: C.paper, opacity: fadeIO(f, PIPELINE, 14), padding: '0 110px', justifyContent: 'center' }}>
      <div style={rise(f, 2)}><Eyebrow>The pipeline</Eyebrow></div>
      <div style={{ ...rise(f, 8), font: `700 76px/1.02 ${F.display}`, color: C.ink, margin: '24px 0 70px' }}>Six stages, two frozen models, one trained verifier</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 22, position: 'relative' }}>
        {STAGES.map((s, i) => {
          const at = 22 + i * 12;
          const k = interpolate(f, [at, at + 16], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease });
          return (
            <div key={s} style={{ opacity: k, transform: `translateY(${(1 - k) * 26}px)`, background: C.sheet, border: `2px solid ${C.ink}`, padding: '24px 22px 26px', minHeight: 250, position: 'relative' }}>
              <div style={{ width: 52, height: 52, display: 'grid', placeItems: 'center', background: C.hi, border: `2px solid ${C.ink}`, font: `600 26px ${F.mono}` }}>{i + 1}</div>
              <div style={{ font: `700 36px/1.05 ${F.display}`, color: C.ink, marginTop: 22 }}>{s}</div>
              <div style={{ font: `400 20px/1.3 ${F.mono}`, color: C.ink3, marginTop: 14 }}>{WHO[i]}</div>
              {i < 5 && <div style={{ position: 'absolute', right: -22, top: 48, width: 22, height: 2, background: C.ink }} />}
            </div>
          );
        })}
      </div>
      <div style={{ ...rise(f, 100), font: `400 28px/1.45 ${F.serif}`, color: C.ink2, marginTop: 56, maxWidth: 1500 }}>
        Next: the project website stepping through these stages for this question. Everything on screen is the saved output of the frozen system.
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ 4. the screen recording with labels
export const FRAME = { x: 240, y: 150, w: 1440, h: 810 };       // the browser frame; recording is 1920x1080
const BASE = FRAME.w / rec.width;                         // 0.75

// The page changes stage at the click, logged a moment later (record.mjs waits after each click),
// so the tracker and caption switch this much earlier for those moments.
const LEAD: Record<string, number> = { s2: 0.55, s3: 0.35, s4: 0.45, s5: 0.35, s6: 0.45 };
const labelAt = (t: number) => {
  let i = 0;
  for (let k = 0; k < rec.moments.length; k++) if (rec.moments[k].t - (LEAD[rec.moments[k].id] ?? 0) <= t) i = k;
  return i;
};
const momentAt = (t: number) => {
  let i = 0;
  for (let k = 0; k < rec.moments.length; k++) if (rec.moments[k].t <= t) i = k;
  return i;
};
const camOf = (i: number) => {
  const m = rec.moments[i], b = BEATS[m.id];
  const box = b?.focus ? (m.boxes as Record<string, Box | null>)[b.focus] : null;
  const s = b?.zoom ?? 1;
  const cx = box ? box[0] + box[2] / 2 : rec.width / 2, cy = box ? box[1] + box[3] / 2 : rec.height / 2;
  return { s, cx, cy };
};
const place = (c: { s: number; cx: number; cy: number }) => {
  const sc = BASE * c.s;
  const tx = Math.min(0, Math.max(FRAME.w - rec.width * sc, FRAME.w / 2 - c.cx * sc));
  const ty = Math.min(0, Math.max(FRAME.h - rec.height * sc, FRAME.h / 2 - c.cy * sc));
  return { sc, tx, ty };
};

export const Highlight: React.FC<{ box: Box; label: string; age: number; left: number; dim: boolean }> = ({ box, label, age, left, dim }) => {
  const k = interpolate(age, [S(0.25), S(0.65)], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease });
  const out = interpolate(left, [0, S(0.3)], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const [x, y, w, h] = box, pad = 10;
  const y0 = Math.max(4, y - pad), y1 = Math.min(rec.height - 4, y + h + pad);
  const labelAbove = y0 > 60;
  return (
    <div style={{ position: 'absolute', left: x - pad, top: y0, width: w + 2 * pad, height: y1 - y0, opacity: k * out,
      transform: `scale(${1.03 - 0.03 * k})`, transformOrigin: 'center' }}>
      <div style={{ position: 'absolute', inset: 0, border: `4px solid ${C.hi}`, borderRadius: 6,
        boxShadow: `0 0 0 2px ${C.ink}${dim ? `, 0 0 0 9999px rgba(23,21,15,${0.1 * k * out})` : ''}` }} />
      <div style={{ position: 'absolute', left: -2, ...(labelAbove ? { bottom: '100%', marginBottom: 8 } : { top: '100%', marginTop: 8 }),
        background: C.ink, color: C.hi, font: `600 25px ${F.mono}`, letterSpacing: '0.02em', padding: '8px 14px', whiteSpace: 'nowrap' }}>{label}</div>
    </div>
  );
};

export const StageTracker: React.FC<{ stage: number; f: number }> = ({ stage }) => (
  <div style={{ position: 'absolute', left: FRAME.x, top: 40, width: FRAME.w, display: 'flex', alignItems: 'center', gap: 10 }}>
    {STAGES.map((s, i) => {
      const on = i + 1 === stage, done = i + 1 < stage;
      return (
        <div key={s} style={{ flex: on ? 2.3 : 1, display: 'flex', alignItems: 'center', gap: 10, height: 64, padding: '0 14px',
          background: on ? C.hi : done ? C.sheet : 'transparent', border: `2px solid ${on || done ? C.ink : C.rule}`, overflow: 'hidden' }}>
          <span style={{ font: `600 21px ${F.mono}`, color: on || done ? C.ink : C.ink3 }}>{i + 1}</span>
          {on
            ? <span style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.05 }}>
                <span style={{ font: `700 27px ${F.display}`, color: C.ink, whiteSpace: 'nowrap' }}>{s}</span>
                <span style={{ font: `500 16px ${F.mono}`, color: C.ink2, whiteSpace: 'nowrap', marginTop: 3 }}>{WHO[i]}</span>
              </span>
            : <span style={{ font: `600 21px ${F.ui}`, color: done ? C.ink : C.ink3, whiteSpace: 'nowrap' }}>{SHORT[i]}</span>}
        </div>
      );
    })}
  </div>
);

const Recording: React.FC = () => {
  const f = useCurrentFrame();
  const t = f / FPS;
  const i = momentAt(t), m = rec.moments[i], beat = BEATS[m.id] ?? { stage: 0, caption: '' };
  const li = labelAt(t), label = BEATS[rec.moments[li].id] ?? { stage: 0, caption: '' };
  const labelStart = S(rec.moments[li].t - (LEAD[rec.moments[li].id] ?? 0));
  const start = S(m.t), next = i + 1 < rec.moments.length ? S(rec.moments[i + 1].t) : REC_LEN;
  // camera: ease from the previous moment's framing to this one
  const from = place(camOf(Math.max(0, i - 1))), to = place(camOf(i));
  const k = spring({ frame: f - start, fps: FPS, config: { damping: 200 }, durationInFrames: S(1.1) });
  const cam = { sc: from.sc + (to.sc - from.sc) * k, tx: from.tx + (to.tx - from.tx) * k, ty: from.ty + (to.ty - from.ty) * k };
  const capK = interpolate(f - labelStart, [0, S(0.35)], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease });
  const enter = interpolate(f, [0, 16], [0, 1], { extrapolateRight: 'clamp', easing: ease });
  const leave = interpolate(f, [REC_LEN - 14, REC_LEN], [1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return (
    <AbsoluteFill style={{ background: C.paper, opacity: leave }}>
      {label.stage > 0
        ? <StageTracker stage={label.stage} f={f} />
        : <div style={{ position: 'absolute', left: FRAME.x, top: 50 }}><Eyebrow>The demo website</Eyebrow></div>}
      {/* browser frame */}
      <div style={{ position: 'absolute', left: FRAME.x, top: FRAME.y - 44, width: FRAME.w, opacity: enter,
        transform: `translateY(${(1 - enter) * 30}px)`, border: `2px solid ${C.ink}`, boxShadow: `12px 12px 0 ${C.hi}`, background: C.sheet }}>
        <div style={{ height: 42, display: 'flex', alignItems: 'center', gap: 9, padding: '0 16px', borderBottom: `2px solid ${C.ink}` }}>
          {[0, 1, 2].map(n => <span key={n} style={{ width: 13, height: 13, borderRadius: 7, border: `2px solid ${C.ink}`, background: n === 0 ? C.hi : 'transparent' }} />)}
          <span style={{ marginLeft: 18, font: `500 19px ${F.mono}`, color: C.ink2 }}>localhost · mmJEE Reasoner demo website</span>
        </div>
        <div style={{ position: 'relative', width: FRAME.w, height: FRAME.h, overflow: 'hidden' }}>
          <div style={{ position: 'absolute', left: 0, top: 0, width: rec.width, height: rec.height, transformOrigin: '0 0',
            transform: `translate(${cam.tx}px, ${cam.ty}px) scale(${cam.sc})` }}>
            <OffthreadVideo src={staticFile('recording.mp4')} style={{ width: rec.width, height: rec.height }} />
            {(beat.hl ?? []).map((h, n) => {
              const box = (m.boxes as Record<string, Box | null>)[h.key];
              if (!box) return null;
              const end = h.dur ? Math.min(next, start + S(h.dur)) : next;
              if (f >= end) return null;
              return <Highlight key={h.key} box={box} label={h.label} age={f - start} left={end - f} dim={n === 0} />;
            })}
          </div>
        </div>
      </div>
      {/* caption */}
      {label.caption && (
        <div style={{ position: 'absolute', left: FRAME.x, top: FRAME.y + FRAME.h + 26, width: FRAME.w, display: 'flex', alignItems: 'stretch',
          opacity: capK, transform: `translateY(${(1 - capK) * 12}px)` }}>
          <div style={{ width: 12, background: C.hi, flexShrink: 0 }} />
          <div style={{ background: C.ink, color: C.sheet, padding: '14px 24px', font: `400 27px/1.32 ${F.serif}`, flex: 1 }}>{label.caption}</div>
        </div>
      )}
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ 5. results
const Results: React.FC = () => {
  const f = useCurrentFrame();
  const grow = (at: number) => interpolate(f, [at, at + 30], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease });
  const bars: [string, number, string][] = [
    ['One chain-of-thought answer', 44.3, C.ink3],
    ['Majority vote, same token budget (18 answers)', 52.5, C.ink2],
    ['Our framework', 58.8, C.hi],
  ];
  return (
    <AbsoluteFill style={{ background: C.paper, opacity: fadeIO(f, RESULTS, 14), padding: '0 150px', justifyContent: 'center' }}>
      <div style={rise(f, 2)}><Eyebrow>Results on all 204 questions of 2026</Eyebrow></div>
      <div style={{ ...rise(f, 8), font: `700 80px/1.0 ${F.display}`, color: C.ink, margin: '24px 0 56px' }}>
        Better than voting at the <Mark>same compute</Mark>, on an exam the model never saw
      </div>
      {bars.map(([label, v, col], i) => (
        <div key={label} style={{ ...rise(f, 18 + i * 8), display: 'grid', gridTemplateColumns: '560px 1fr 130px', alignItems: 'center', gap: 28, padding: '16px 0', borderTop: `1px solid ${C.rule}` }}>
          <span style={{ font: `${i === 2 ? 700 : 500} 28px ${F.ui}`, color: C.ink }}>{label}</span>
          <div style={{ height: 34, background: '#ebe5d3', position: 'relative' }}>
            <div style={{ position: 'absolute', inset: 0, width: `${(v / 70) * 100 * grow(24 + i * 8)}%`, background: col,
              boxShadow: i === 2 ? `inset 0 0 0 2px ${C.ink}` : 'none' }} />
          </div>
          <span style={{ font: `700 46px ${F.display}`, color: C.ink, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{v.toFixed(1)}%</span>
        </div>
      ))}
      <div style={{ ...rise(f, 60), font: `400 28px/1.45 ${F.serif}`, color: C.ink2, marginTop: 40, maxWidth: 1550 }}>
        +6.4 points over majority voting with the same token budget (95% CI +1.5 to +10.8, McNemar p = 0.011). The system was frozen on 2019–2024 data before any 2026 run.
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ 6. end card
export const End: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: C.ink, opacity: interpolate(f, [0, 14], [0, 1], { extrapolateRight: 'clamp' }), padding: '0 150px', justifyContent: 'center' }}>
      <div style={{ ...rise(f, 4), font: `700 110px/1.0 ${F.display}`, color: C.sheet }}>mmJEE <span style={{ color: C.hi }}>Reasoner</span></div>
      <div style={{ ...rise(f, 12), font: `500 32px ${F.ui}`, color: '#c2bba8', marginTop: 30 }}>Team Erdős · Arpit Kumar (12340350) · Saurav Gupta (12341940) · IIT Bhilai</div>
      <div style={{ ...rise(f, 20), font: `500 28px ${F.mono}`, color: C.hi, marginTop: 44 }}>github.com/Saurav1375/erdos-mmjee-reasoner</div>
    </AbsoluteFill>
  );
};

export const MmjeeDemo: React.FC = () => (
  <AbsoluteFill style={{ background: C.paper }}>
    <Sequence from={T_TITLE} durationInFrames={TITLE}><Title /></Sequence>
    <Sequence from={T_Q} durationInFrames={QUESTION}><Question /></Sequence>
    <Sequence from={T_PIPE} durationInFrames={PIPELINE}><Pipeline /></Sequence>
    <Sequence from={T_REC} durationInFrames={REC_LEN}><Recording /></Sequence>
    <Sequence from={T_RES} durationInFrames={RESULTS}><Results /></Sequence>
    <Sequence from={T_END} durationInFrames={END}><End /></Sequence>
  </AbsoluteFill>
);
