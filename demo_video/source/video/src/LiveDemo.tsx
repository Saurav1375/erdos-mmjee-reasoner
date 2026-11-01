import React from 'react';
import { AbsoluteFill, Img, OffthreadVideo, Sequence, staticFile, interpolate, useCurrentFrame } from 'remotion';
import { C, F, S, FPS, STAGES, ease, fadeIO, rise, Eyebrow, Mark, FRAME, Highlight, StageTracker, End } from './Demo';
import mom from './live_moments.json';
import evs from './live_events.json';
import map from './live_map.json';
import q from './live_question.json';

// ------------------------------------------------------------------ data from the recorded run
type Box = [number, number, number, number];
type Ev = { t: number; type: string; i?: number; status?: string; round?: number; n?: number;
  retrieval?: [string, number][]; chains?: { s: number; error?: boolean; step?: number | null; answer?: string }[];
  final?: { pred: string | null; ok: boolean } };
const E = evs as Ev[];
const M = mom.moments as { id: string; t: number; boxes: Record<string, Box | null> }[];
const T = (id: string) => M.find(m => m.id === id)?.t ?? 0;
const GO = T('go');
const SEGS = map.segments as { src: [number, number]; out: [number, number]; speed: number }[];
const srcAt = (o: number) => {                    // edited time -> recording time
  const s = SEGS.find(g => o < g.out[1]) ?? SEGS[SEGS.length - 1];
  return { t: s.src[0] + (Math.min(o, s.out[1]) - s.out[0]) * s.speed, speed: s.speed, seg: s };
};
const done = E.find(e => e.type === 'done');
const retrieval = E.find(e => e.type === 'stage' && e.i === 1 && e.status === 'done')?.retrieval ?? [];
const critics = E.filter(e => e.type === 'critic');
const correctors = E.filter(e => e.type === 'corrector');
const realSecs = (done?.t ?? GO) - GO;
const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const sims = retrieval.map(r => r[1].toFixed(3)).join(' and ');
const exNames = retrieval.map(([id]) => id.startsWith('jb') ? `JEEBench ${id.slice(2, 6)}` : `JEE Advanced ${id.slice(0, 4)}`);

// ------------------------------------------------------------------ timeline
const TITLE = S(5.5), SETUP = S(8), QUESTION = S(6.5);
const REC_LEN = Math.round(map.length * FPS), RESULT = S(7), END = S(5);
const T_SETUP = TITLE, T_Q = T_SETUP + SETUP, T_REC = T_Q + QUESTION, T_RES = T_REC + REC_LEN, T_END = T_RES + RESULT;
export const LIVE_TOTAL = T_END + END;

// ------------------------------------------------------------------ what is said when (recording time)
const flagged = (r: number) => critics[r]?.chains?.filter(c => c.error).length ?? 0;
const checked = (r: number) => critics[r]?.chains?.length ?? 0;
function captionAt(t: number): string {
  if (t < T('choose')) return 'The live section of the demo website. It sends the question to the two models on the lab GPUs.';
  if (t < T('go')) return `We pick ${q.name} and keep the Full setting: 8 attempts and up to 3 correction rounds, the frozen configuration.`;
  if (t < T('panel_read') + 0.2) return 'Run live. From here everything is computed now, on the lab GPUs and this laptop.';
  if (t < T('panel_look')) return 'Stage 1 · Qwen3-VL-8B reads the question image and writes it out as text, including what the figure shows.';
  if (t < T('panel_solve')) return `Stage 2 · The model tags the question by topic and method. Retrieval finds ${retrieval.length} solved problem${retrieval.length === 1 ? '' : 's'} above the threshold τ = 0.882 (similarity ${sims}). They go into the prompt as worked examples.`;
  if (t < T('ff_solve')) return 'Stage 3 · Eight attempts are written in parallel. Attempt 1 streams here token by token; formulas render once it finishes, and each Python block runs in the sandbox on this laptop.';
  if (t < T('solve_done')) return 'Waiting for all 8 attempts to finish.';
  if (t < T('ff_critic')) return 'All 8 attempts are done. Each chip shows that attempt’s final answer.';
  if (t < T('verify')) {
    const r = critics.filter(c => c.t <= t).length;
    if (r === 0) return 'Stage 4 · InternVL3.5-8B, on the other lab PC, checks each solution for its first wrong step.';
    const k = r - 1, fixed = correctors[k] && correctors[k].t <= t ? ` The solver redoes ${correctors[k].chains?.length ?? 0} of them from the flagged step.` : '';
    return `Stage 4 · Critic round ${k + 1}: ${flagged(k)} of ${checked(k)} solutions flagged.${fixed}`;
  }
  if (t < T('done')) return 'Stage 5 · The XGBoost verifier on this laptop scores all 16 answers (8 original, 8 corrected) and picks one.';
  if (t < T('replay')) return `Done in ${mmss(realSecs)} of real time. Final answer ${done?.final?.pred}, ${done?.final?.ok ? 'correct' : 'wrong'} by the benchmark's scorer.`;
  if (t < T('replay_look')) return 'The finished live run opens in the replay viewer, like any saved run.';
  if (t < T('replay_verify')) return `The ${retrieval.length} solved problem${retrieval.length === 1 ? '' : 's'} retrieved in this live run.`;
  if (t < T('replay_result')) return 'The verifier’s scores for the 16 answers of this live run.';
  return 'The result of the live run, next to the baselines from the saved frozen run.';
}
function stageAt(t: number): number {
  if (t >= T('replay')) return 0;
  if (t >= T('panel_look') && t < T('panel_solve')) return 2;   // retrieval is on screen until the view moves on
  const st = E.filter(e => e.type === 'stage' && e.t <= t && e.status === 'start').pop();
  if (done && t >= done.t) return 6;
  return st ? (st.i ?? 0) + 1 : 0;
}
const HL: Record<string, { key: string; label: string; dur: number }[]> = {
  choose: [{ key: 'select', label: 'Question', dur: 3.0 }],
  panel_read: [{ key: 'panel', label: 'Transcript, written live', dur: 2.6 }],
  look_view: [{ key: 'panel', label: `Concept tag · ${retrieval.length} solved examples retrieved`, dur: 4.05 }],
  done: [{ key: 'final', label: 'Final answer of the live run', dur: 6.0 }],
  replay_look: [{ key: 'examples', label: 'Retrieved in this live run', dur: 4.0 }],
  replay_verify: [{ key: 'pick', label: "Verifier's pick", dur: 4.0 }],
  replay_result: [{ key: 'ours', label: 'Live run', dur: 4.6 }],
};

// ------------------------------------------------------------------ cards
const Title: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: C.paper, opacity: fadeIO(f, TITLE, 14), padding: '0 150px', justifyContent: 'center' }}>
      <div style={rise(f, 4)}><Eyebrow>mmJEE Reasoner · live demo</Eyebrow></div>
      <div style={{ ...rise(f, 12, 30), font: `700 112px/1.0 ${F.display}`, color: C.ink, marginTop: 34, maxWidth: 1500 }}>
        Solving a JEE question <Mark>live</Mark> on the lab GPUs
      </div>
      <div style={{ ...rise(f, 24), font: `400 34px/1.45 ${F.serif}`, color: C.ink2, marginTop: 40, maxWidth: 1350 }}>
        The same frozen system as in the paper, run on demand. Waiting time is fast-forwarded and marked on screen; the real elapsed time is shown throughout.
      </div>
      <div style={{ ...rise(f, 34), marginTop: 70, paddingTop: 22, borderTop: `2px solid ${C.ink}`, font: `500 26px ${F.ui}`, color: C.ink2, display: 'flex', gap: 48 }}>
        <span><b style={{ color: C.ink }}>Team Erdős</b></span><span>Arpit Kumar · Saurav Gupta</span><span>IIT Bhilai · Machine Learning course project</span>
      </div>
    </AbsoluteFill>
  );
};

const Node: React.FC<{ head: string; sub: string; lines: string[]; at: number; f: number; hi?: boolean }> = ({ head, sub, lines, at, f, hi }) => (
  <div style={{ ...rise(f, at, 26), background: hi ? C.hi : C.sheet, border: `2px solid ${C.ink}`, padding: '26px 28px', boxShadow: hi ? 'none' : `10px 10px 0 ${C.hi}` }}>
    <div style={{ font: `700 44px/1 ${F.display}`, color: C.ink }}>{head}</div>
    <div style={{ font: `500 21px ${F.mono}`, color: C.ink2, marginTop: 10 }}>{sub}</div>
    <div style={{ marginTop: 22, display: 'grid', gap: 10 }}>
      {lines.map(l => <div key={l} style={{ font: `400 26px/1.3 ${F.serif}`, color: C.ink, paddingTop: 10, borderTop: `1px solid ${hi ? C.ink : C.rule}` }}>{l}</div>)}
    </div>
  </div>
);
const Setup: React.FC = () => {
  const f = useCurrentFrame();
  const line = (at: number) => interpolate(f, [at, at + 20], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease });
  return (
    <AbsoluteFill style={{ background: C.paper, opacity: fadeIO(f, SETUP, 14), padding: '0 110px', justifyContent: 'center' }}>
      <div style={rise(f, 2)}><Eyebrow>The live setup</Eyebrow></div>
      <div style={{ ...rise(f, 8), font: `700 74px/1.02 ${F.display}`, color: C.ink, margin: '24px 0 60px' }}>Two lab PCs for the models, the laptop for everything else</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 150px 1fr', alignItems: 'center' }}>
        <Node f={f} at={20} head="Laptop" sub="browser + demo server" lines={['Code sandbox (runs the model’s Python)', 'Retrieval: bge-small + FAISS, 606 solved problems', 'Verifier: XGBoost ranker over 16 answers']} />
        <div style={{ position: 'relative', height: 260 }}>
          {[70, 190].map((y, k) => (
            <div key={y} style={{ position: 'absolute', left: 10, top: y, width: `${130 * line(40 + k * 8)}px`, height: 3, background: C.ink }}>
              <div style={{ position: 'absolute', right: -2, top: -7, width: 0, height: 0, borderLeft: `14px solid ${C.ink}`, borderTop: '8px solid transparent', borderBottom: '8px solid transparent', opacity: line(40 + k * 8) }} />
            </div>
          ))}
          <div style={{ position: 'absolute', left: 0, right: 0, top: 108, textAlign: 'center', font: `500 18px ${F.mono}`, color: C.ink3, opacity: line(46) }}>SSH tunnels</div>
        </div>
        <div style={{ display: 'grid', gap: 26 }}>
          <Node f={f} at={30} hi head="ml-lab2 · RTX 3060 12 GB" sub="vllm serve · Qwen3-VL-8B (4-bit)" lines={['Transcript, tag, 8 attempts, corrections']} />
          <Node f={f} at={38} hi head="ml-lab · RTX 3060 12 GB" sub="vllm serve · InternVL3.5-8B (4-bit)" lines={['Critic: finds the first wrong step']} />
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Question: React.FC = () => {
  const f = useCurrentFrame();
  const rows: [string, string][] = [['Exam', q.exam], ['Subject', q.subject], ['Language', 'English'], ['Answer key', q.gold]];
  return (
    <AbsoluteFill style={{ background: C.paper, opacity: fadeIO(f, QUESTION, 14), padding: '0 120px', justifyContent: 'center' }}>
      <div style={rise(f, 4)}><Eyebrow>The question</Eyebrow></div>
      <div style={{ ...rise(f, 10), font: `700 70px/1.02 ${F.display}`, color: C.ink, margin: '24px 0 40px', maxWidth: 1500 }}>{q.headline}</div>
      <div style={{ display: 'flex', gap: 70, alignItems: 'flex-start' }}>
        <div style={{ ...rise(f, 16, 30), width: 980, flexShrink: 0, background: '#fff', border: `2px solid ${C.ink}`, boxShadow: `14px 14px 0 ${C.hi}`, padding: 18 }}>
          <Img src={staticFile('question_live.webp')} style={{ width: '100%', display: 'block' }} />
        </div>
        <div style={{ flex: 1 }}>
          {rows.map(([k, v], i) => (
            <div key={k} style={{ ...rise(f, 22 + i * 5), display: 'grid', gridTemplateColumns: '200px 1fr', padding: '14px 0', borderTop: `1px solid ${C.rule}`, font: `400 27px/1.35 ${F.serif}`, color: C.ink }}>
              <span style={{ font: `500 20px ${F.mono}`, letterSpacing: '0.06em', textTransform: 'uppercase', color: C.ink3, paddingTop: 5 }}>{k}</span>
              <span style={k === 'Answer key' ? { font: `600 30px ${F.mono}` } : {}}>{v}</span>
            </div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ the edited live recording
const BASE = FRAME.w / mom.width;
const Recording: React.FC = () => {
  const f = useCurrentFrame();
  const { t, speed, seg } = srcAt(f / FPS);
  const stage = stageAt(t);
  const real = Math.max(0, Math.min(t, done?.t ?? t) - GO);
  const ff = speed > 1, paused = speed === 0;
  const ffK = interpolate(f / FPS, [seg.out[0], seg.out[0] + 0.3, seg.out[1] - 0.3, seg.out[1]], [0, 1, 1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const caption = captionAt(t);
  const capStart = Math.max(...M.filter(m => m.t <= t).map(m => m.t), 0);
  const enter = interpolate(f, [0, 16], [0, 1], { extrapolateRight: 'clamp', easing: ease });
  const leave = interpolate(f, [REC_LEN - 14, REC_LEN], [1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const cur = M.filter(m => m.t <= t).pop();
  return (
    <AbsoluteFill style={{ background: C.paper, opacity: leave }}>
      {stage > 0
        ? <StageTracker stage={stage} f={f} />
        : <div style={{ position: 'absolute', left: FRAME.x, top: 50 }}><Eyebrow>{t >= T('replay') ? 'The live run, opened in the replay viewer' : 'Live run · lab GPUs'}</Eyebrow></div>}
      <div style={{ position: 'absolute', left: FRAME.x, top: FRAME.y - 44, width: FRAME.w, opacity: enter,
        transform: `translateY(${(1 - enter) * 30}px)`, border: `2px solid ${C.ink}`, boxShadow: `12px 12px 0 ${C.hi}`, background: C.sheet }}>
        <div style={{ height: 42, display: 'flex', alignItems: 'center', gap: 9, padding: '0 16px', borderBottom: `2px solid ${C.ink}` }}>
          {[0, 1, 2].map(n => <span key={n} style={{ width: 13, height: 13, borderRadius: 7, border: `2px solid ${C.ink}`, background: n === 0 ? C.hi : 'transparent' }} />)}
          <span style={{ marginLeft: 18, font: `500 19px ${F.mono}`, color: C.ink2 }}>localhost:8765 · live mode · lab connected</span>
          {t >= GO && <span style={{ marginLeft: 'auto', font: `600 20px ${F.mono}`, color: C.ink }}>real time {mmss(real)}</span>}
        </div>
        <div style={{ position: 'relative', width: FRAME.w, height: FRAME.h, overflow: 'hidden' }}>
          <div style={{ position: 'absolute', left: 0, top: 0, width: mom.width, height: mom.height, transformOrigin: '0 0', transform: `scale(${BASE})` }}>
            <OffthreadVideo src={staticFile('live_edit.mp4')} style={{ width: mom.width, height: mom.height }} />
            {!ff && cur && (HL[cur.id] ?? []).map((h, n) => {
              const box = cur.boxes[h.key]; if (!box) return null;
              const age = (t - cur.t) * FPS, left = (h.dur - (t - cur.t)) * FPS;
              if (left <= 0) return null;
              return <Highlight key={h.key} box={box} label={h.label} age={age} left={left} dim={n === 0} />;
            })}
          </div>
          {paused && (
            <div style={{ position: 'absolute', right: 26, top: 22, opacity: ffK, display: 'flex', alignItems: 'stretch', boxShadow: `6px 6px 0 ${C.ink}` }}>
              <div style={{ background: C.sheet, border: `2px solid ${C.ink}`, padding: '10px 18px', font: `700 40px/1 ${F.display}`, color: C.ink }}>❚❚</div>
              <div style={{ background: C.ink, padding: '10px 18px', display: 'flex', alignItems: 'center', font: `600 17px ${F.mono}`, color: C.hi, letterSpacing: '0.06em' }}>PAUSED TO READ</div>
            </div>
          )}
          {ff && (
            <div style={{ position: 'absolute', right: 26, top: 22, opacity: ffK, display: 'flex', alignItems: 'stretch', boxShadow: `6px 6px 0 ${C.ink}` }}>
              <div style={{ background: C.hi, border: `2px solid ${C.ink}`, padding: '10px 20px', font: `700 46px/1 ${F.display}`, color: C.ink }}>▶▶ {Math.round(speed)}×</div>
              <div style={{ background: C.ink, padding: '10px 18px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <span style={{ font: `600 17px ${F.mono}`, color: C.hi, letterSpacing: '0.06em' }}>FAST-FORWARD</span>
                <span style={{ font: `500 17px ${F.mono}`, color: '#c2bba8' }}>{mmss(seg.src[1] - seg.src[0])} of waiting shown in {Math.round(seg.out[1] - seg.out[0])} s</span>
              </div>
            </div>
          )}
        </div>
      </div>
      <div style={{ position: 'absolute', left: FRAME.x, top: FRAME.y + FRAME.h + 26, width: FRAME.w, display: 'flex', alignItems: 'stretch' }}>
        <div style={{ width: 12, background: C.hi, flexShrink: 0 }} />
        <div key={caption} style={{ background: C.ink, color: C.sheet, padding: '14px 24px', font: `400 26px/1.32 ${F.serif}`, flex: 1,
          opacity: interpolate((t - capStart) * FPS, [0, 8], [0.4, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }) }}>{caption}</div>
      </div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------ live result card
const Result: React.FC = () => {
  const f = useCurrentFrame();
  const flaggedTotal = critics.reduce((n, c) => n + (c.chains?.filter(x => x.error).length ?? 0), 0);
  const corrections = correctors.reduce((n, c) => n + (c.chains?.length ?? 0), 0);
  const stats: [string, string][] = [
    ['Real time', `${mmss(realSecs)}`],
    ['Solved examples retrieved', `${retrieval.length} · ${exNames.join(', ')}`],
    ['Attempts', '8, with Python run in the sandbox'],
    ['Critic flags / corrections', `${flaggedTotal} / ${corrections} over ${critics.length} round${critics.length === 1 ? '' : 's'}`],
    ['Final answer', `${done?.final?.pred ?? '(none)'} · answer key ${q.gold}`],
  ];
  return (
    <AbsoluteFill style={{ background: C.paper, opacity: fadeIO(f, RESULT, 14), padding: '0 150px', justifyContent: 'center' }}>
      <div style={rise(f, 2)}><Eyebrow>This live run</Eyebrow></div>
      <div style={{ ...rise(f, 8), font: `700 86px/1.0 ${F.display}`, color: C.ink, margin: '24px 0 46px' }}>
        {done?.final?.ok ? <>Answered <Mark>correctly</Mark>, from scratch, in {mmss(realSecs)}</> : <>Answered in {mmss(realSecs)}</>}
      </div>
      {stats.map(([k, v], i) => (
        <div key={k} style={{ ...rise(f, 16 + i * 5), display: 'grid', gridTemplateColumns: '460px 1fr', padding: '15px 0', borderTop: `1px solid ${C.rule}` }}>
          <span style={{ font: `500 22px ${F.mono}`, letterSpacing: '0.05em', textTransform: 'uppercase', color: C.ink3, paddingTop: 4 }}>{k}</span>
          <span style={{ font: `${i === 4 ? 600 : 400} 30px ${i === 4 ? F.mono : F.serif}`, color: C.ink }}>{v}</span>
        </div>
      ))}
      <div style={{ ...rise(f, 46), font: `400 25px/1.45 ${F.serif}`, color: C.ink2, marginTop: 30, maxWidth: 1500 }}>
        Live runs draw new samples, so their answers can differ from the saved frozen run. The headline 58.8% on 2026 comes from the frozen runs.
      </div>
    </AbsoluteFill>
  );
};

export const LiveDemo: React.FC = () => (
  <AbsoluteFill style={{ background: C.paper }}>
    <Sequence durationInFrames={TITLE}><Title /></Sequence>
    <Sequence from={T_SETUP} durationInFrames={SETUP}><Setup /></Sequence>
    <Sequence from={T_Q} durationInFrames={QUESTION}><Question /></Sequence>
    <Sequence from={T_REC} durationInFrames={REC_LEN}><Recording /></Sequence>
    <Sequence from={T_RES} durationInFrames={RESULT}><Result /></Sequence>
    <Sequence from={T_END} durationInFrames={END}><End /></Sequence>
  </AbsoluteFill>
);
