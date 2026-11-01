"""Cuts live.mp4 into live_edit.mp4: normal speed, except the two waits logged by live.mjs
(ff_solve -> solve_done: the 8 attempts being written; ff_critic -> verify: the critic and corrector
rounds), which are sped up to last about TARGET seconds each. Writes live_map.json, the segment
list that maps edited time back to recording time, for the overlays in Remotion.

usage: python3 edit_live.py
"""
import json
import subprocess

TARGET = 7.0          # seconds on screen for each fast-forwarded wait
MIN_SPEED = 4.0
HOLDS = [(21.4, 3.0)]  # (recording time, seconds): freeze on the retrieval result so it can be read

m = {x["id"]: x["t"] for x in json.load(open("live_moments.json"))["moments"]}
end = json.load(open("live_moments.json"))["moments"][-1]["t"] + 0.5
waits = [(m["ff_solve"], m["solve_done"]), (m["ff_critic"], m["verify"])] if "ff_critic" in m else [(m["ff_solve"], m["solve_done"])]

cuts = sorted([(a, b, max(MIN_SPEED, round((b - a) / TARGET, 1))) for a, b in waits] + [(h, h, 0.0) for h, _ in HOLDS])
hold_len = dict(HOLDS)
segs, cur = [], 0.0
for a, b, sp in cuts:
    if a > cur:
        segs.append({"src": [cur, a], "speed": 1.0})
    segs.append({"src": [a, b], "speed": sp} | ({"hold": hold_len[a]} if sp == 0 else {}))
    cur = b
segs.append({"src": [cur, end], "speed": 1.0})

out = 0.0
for s in segs:
    n = s["hold"] if s["speed"] == 0 else (s["src"][1] - s["src"][0]) / s["speed"]
    s["out"] = [round(out, 3), round(out + n, 3)]
    out = s["out"][1]

parts, labels = [], []
for k, s in enumerate(segs):
    a, b = s["src"]
    if s["speed"] == 0:      # freeze one frame
        n = round(s["hold"] * 30)
        parts.append(f"[0:v]trim=start={a}:duration=0.034,setpts=PTS-STARTPTS,fps=30,loop=loop={n - 1}:size=1:start=0,setpts=N/30/TB[v{k}]")
    else:
        parts.append(f"[0:v]trim=start={a}:end={b},setpts=(PTS-STARTPTS)/{s['speed']},fps=30[v{k}]")
    labels.append(f"[v{k}]")
graph = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(segs)}:v=1:a=0[out]"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "live.mp4", "-filter_complex", graph, "-map", "[out]",
                "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "live_edit.mp4"], check=True)
json.dump({"segments": segs, "length": round(out, 3)}, open("live_map.json", "w"), indent=1)
for s in segs:
    print(f"src {s['src'][0]:7.1f}-{s['src'][1]:7.1f}  x{s['speed']:<5}  out {s['out'][0]:6.1f}-{s['out'][1]:6.1f}")
print(f"edited length {out:.1f} s")
