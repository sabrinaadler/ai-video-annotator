# Complete Fieldnote workflow preview

The chosen Field instrument direction now covers five stages:

1. Open the prepared mouse054 recording and inspect its metadata.
2. Define the target behavior, session definition, and numeric shortcut.
3. Mark example intervals or use the five prepared examples.
4. Review cached suggestions with comparison, boundary edits, notes, accept,
   reject, later, and undo/redo.
5. Preview annotation CSV and a session JSON record, or return to unresolved work.

This is an in-conversation prototype, not a production frontend. It uses real
recorded snapshots and the fixed full-frame five-example retrieval result.
Changing examples does not rerun inference. Recording import, playback, frame
stepping, and search are represented rather than connected to a live decoder.
No source videos, annotations, or embedding artifacts are modified.

## Session and export behavior

Examples must exist before review can start. Behavior names, reserved shortcut
collisions, invalid temporal bounds, and duplicate examples are validated.
Backtracking preserves examples and review decisions. Accept/reject/later advance
to the next pending suggestion; undo restores the prior decision and selection.
Only accepted suggestions become annotations. Deferred candidates are not negative
examples. Session JSON retains all statuses and notes, including unresolved work.

Annotation CSV includes manually added labels and accepted suggestions. Inclusion
of example intervals is configurable. Rejected, deferred, and unreviewed candidates
are excluded. CSV quotes embedded punctuation/newlines and prefixes formula-like
text. Rows identify the video, behavior, interval in seconds, provenance, and note.

The export buttons display selectable CSV/JSON text inside the preview. They do
not claim to create files through the embedded surface. The conversation host may
save preview state; this is not production application autosave. Snapshots stay
below the host size limit, with an explicit session-only fallback when exceeded.

## Verification

An isolated Chrome test passed a complete session: recording selection, invalid
name and shortcut rejection, invalid/duplicate interval rejection, five-example
selection, cached review, acceptance, deferral, rejection, undo, export contents,
and return to unresolved work. It confirmed that unresolved suggestions never
appear in annotation CSV but remain in session JSON, and that backtracking retains
example intervals. No runtime errors or horizontal overflow were found across
all five stages at widths of 1024, 736, and 360 pixels. The opening, examples,
and export screens were inspected visually.

The next user validation task is to complete this flow with minimal explanation:
label a behavior, review and adjust two suggestions, defer an ambiguous case, then
explain what is included in each export format.
