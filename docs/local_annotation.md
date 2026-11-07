# Fieldnote local annotation workspace

Fieldnote now uses real video playback, researcher-authored examples, the frozen VideoMAE
embedding cache, and the existing nearest-example/prototype retrieval implementation.
No annotation labels are loaded from CalMS21 ground truth by the interface.

Launch from the repository:

```sh
PYTHONPATH=src .venv/bin/python scripts/annotate.py \
  --video data/raw/calms21/task1_videos_mp4/train/mouse054_task1_annotator1.mp4 \
  --cache outputs/embeddings/mouse054_videomae_letterbox \
  --session outputs/sessions/fieldnote-mouse054.json
```

Open http://127.0.0.1:8765. The server binds only to loopback. Stop it with Ctrl+C.
The original recording's SHA256 must match the cache before the workspace opens.
CalMS21's MPEG-4 Part 2 recording is converted once to an H.264 viewing copy under
`outputs/viewing`; the original research inputs remain intact. Frame timestamps are
preserved. Reuse verifies both source and derivative hashes. The first launch takes
longer while this copy is prepared. H.264 input recordings are served directly.
PyAV (`ml` extra) is required; no new server framework is needed.

1. Name a behavior and choose nearest-example or mean-prototype retrieval.
2. Seek, play, and mark clear examples with I/O or numeric boundaries. Add up to ten
   distinct examples. At least one example is required. The cache maps each example
   to its most overlapping window; two examples mapping to one window are rejected.
3. Find similar moments. This runs actual vector retrieval and temporal suppression;
   seed intervals and their corresponding cached windows are excluded.
4. Refine suggestion boundaries, write observation notes, and accept, reject, or defer.
   Accept/reject/defer advances to the next pending suggestion. Undo restores the last
   saved edit. Search settings lock after retrieval to preserve decision provenance.
5. Export annotation CSV or the full session JSON. CSV includes researcher examples
   and accepted suggestions; JSON retains rejected, deferred, pending, original
   proposal boundaries, scores, and model/cache provenance.

Every reference/setting/review edit writes an atomic session JSON to the configured
path. Relaunch with the same path to resume. Concurrent stale writes are rejected,
with a reload message. Playback position is saved alongside annotation edits.
Use a different `--session` path for a new behavior or search. Only the configured
recording/cache is available in this first version; file importing and multi-behavior
projects are future work. There is no model training or live video embedding here.
Similarity scores are cosine rankings, not probabilities. Frame stepping advances
by 1/fps; browser seeking is not a substitute for decoder-level frame-accurate export.

Shortcuts: Space play/pause; arrows one frame; I/O boundaries; Enter add example;
A accept; R reject; D defer; N/P suggestions; L loop; Cmd/Ctrl+Z undo; ? help.
Shortcuts do not intercept typing in editable controls.

Verification includes source/cache mismatch handling, actual cache retrieval, atomic
save/resume, stale revision rejection, CSV quoting/formula protection, partial-content
video range serving, local-only writes, and derivative frame timestamp preservation.
Browser checks exercise the real CalMS21 recording and cache using a separate session.

The optional real-browser regression script is `tests/browser/annotation.cjs`.
Start the command above with a fresh temporary `--session` and `--port 8766`, then
run it with Node and Playwright installed. It uses the prepared recording/cache and
five explicit benchmark intervals solely in the test session. It expects macOS
Google Chrome at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`.
Set `FIELDNOTE_TEST_URL` to change the test server URL. Keep this separate from your
research annotation session; the browser test intentionally writes decisions.
