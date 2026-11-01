# mmJEE demo video

A labelled walkthrough of the demo website for one question, P2-HI-Chem-Num-q7. The website replays
the saved run of the frozen system: Hindi image, English transcript, retrieval of two solved
problems, Python in the sandbox, the critic, the verifier's pick, and the result.

Recorded with Playwright (system Chrome) and edited with Remotion. The rendered videos are in `demo_video/`.

## Rebuild

```
# 1. serve the website (from the repo root)
cd demo_website
python3 -m http.server 8810 --bind 127.0.0.1

# 2. record it (Chrome screencast; writes recording/demo.mp4 and recording/moments.json)
cd demo_video/source/recording && npm install && node record.mjs

# 3. copy the recording into the video project and render
cp demo.mp4 ../video/public/recording.mp4 && cp moments.json ../video/src/moments.json
cd ../video && npm install && npm run render        # out/mmjee-demo.mp4
npm run studio                       # optional: live preview
```

## Files

| File | What it does |
|---|---|
| `recording/record.mjs` | Drives the page with Playwright (system Chrome), moves a visible cursor, and records with Chrome's screencast at 1920x1080 (page laid out at 1440x810). It logs the time and on-screen box of every labelled moment to `moments.json`. |
| `video/src/Demo.tsx` | The Remotion edit: title, question card, pipeline overview, the recording in a browser frame with a stage tracker, captions, labelled highlight boxes and zooms, the 2026 results, and an end card. Captions and labels are in `BEATS`. |

To change the question, set `QID` in `record.mjs`, re-record, and update the captions in `BEATS`,
which name the question's facts (similarities, answer, votes).

## Live-run video (`LiveDemo`)

A second video records a real live run on the lab GPUs (P1-EN-Phys-MCQM-q3, English): the
transcript written live, retrieval of two solved problems, 8 attempts streamed, three critic and
corrector rounds, the verifier's pick, and the run opened in the replay viewer. Waits are
fast-forwarded with an on-screen "▶▶ N×" badge and the real elapsed time shown throughout; one
frame is held for 3 s (marked "Paused to read") so the retrieval result can be read.

```
bash demo_website/live/start.sh   # lab models + demo server (localhost:8765)
cd recording && node live.mjs          # records the live run: live.mp4, live_events.json, live_moments.json
python3 edit_live.py                   # fast-forwards the waits: live_edit.mp4 + live_map.json
cp live_edit.mp4 ../video/public/ && cp live_moments.json live_events.json live_map.json ../video/src/
cd ../video && npx remotion render LiveDemo out/mmjee-live-demo.mp4 --browser-executable=/usr/bin/google-chrome-stable --concurrency=2
```

Use a question that has not been run live before: an earlier run's transcript and tag are cached,
and then appear instantly. If retrieval finishes too quickly to be logged as its own moment, add
it to `live_moments.json` by hand (the original is kept as `live_moments.raw.json`).
