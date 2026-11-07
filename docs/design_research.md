# Annotation workspace design research

## Scope and evidence

Desk research conducted 11 October 2026, informed by the user's preference for a
research instrument that is approachable and designed around video. No researcher
interviews or usability study have been conducted. The interface remains a design
preview; there is no new production frontend or API.

The previous preview provided controls but lacked a task hierarchy: the researcher
had to move attention among similarly prominent buttons and panels. That is our
design diagnosis, not a measured user-study finding.

## What existing tools establish

[BORIS coding documentation](https://www.boris.unito.it/user_guide/coding/) supports
behavior shortcuts, one-frame navigation, adjustable playback, event comments, and
undo. The ethogram exposes key bindings rather than hiding them in settings.
Its [preferences](https://www.boris.unito.it/user_guide/preferences/) also support
starting playback before an event and saving work periodically. These are evidence
that timing, context, and recoverability are established annotation needs.

[ELAN's manual](https://www.mpi.nl/tools/elan/docs/manual/oxygen-main.html) organizes
annotation around tiers, selections, time zoom, playback around a selection, and
editable shortcuts. The lesson is to make the selected temporal interval a central
object, not just a pair of numeric fields.

[W3C text contrast](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html)
sets 4.5:1 for normal text. [Use of color](https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html)
requires an additional means of conveying meaning.
[Non-text contrast](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html)
sets 3:1 for necessary visual indications of controls and graphical objects.
These constrain palette use, but do not establish a scientifically “best” hue.

## Design hypotheses to validate

The user's practical questions are: Is this the same behavior? Where does it start
and stop? What did I decide already? How do I return to an ambiguous clip? The screen
should answer those without navigating away from the video.

1. One selected suggestion has the most prominent video, time range, and decision.
2. Before/candidate/after frames and a one-second context loop support comparison.
3. A researcher-labeled reference can be opened alongside the candidate, with its
   timestamp and provenance. We do not offer an invented model explanation.
4. Accept, reject, and later are separate actions. Uncertainty is not automatically
   a negative example. Decisions advance to the next pending suggestion; undo
   restores the previous decision and selection.
5. Manual labels and accepted suggestions belong to the annotation ledger;
   proposals remain hatched and labeled as provisional on the timeline.
6. Keyboard hints sit beside actions. Shortcut handling is scoped to the active
   workspace and disabled inside editable fields.
7. Plain-language labels supplement research terms. Dense controls appear where
   they are useful, without exposing model parameters during review.
8. Ranked similarities are not confidence percentages. This matters especially
   because our actual cosine scores are clustered near 1.0.

## Visual identity

Working name **Fieldnote** is a prototype label, not a final brand decision.

Recommended direction: mineral paper, forest ink, amber for unresolved suggestions.
This borrows the visual precision of a field notebook and laboratory readout while
keeping the video on a darker matte surface. It uses a restrained serif for the
observation title, IBM Plex Sans for controls, and IBM Plex Mono for times and frames.
There are no gradient hero sections or decorative statistic cards.

| Role | Light appearance | Dark appearance |
|---|---|---|
| Workspace | `#F1F0E9` | `#1C2422` |
| Surface | `#FAF9F4` | `#242E2B` |
| Primary text | `#253E36` | `#E3E9DF` |
| Secondary text | `#59695E` | `#B0BEB5` |
| Action / accepted labels | `#285846` | `#A4CDB4` |
| Text on action | `#FBFAF3` | `#18392B` |
| Pending proposal text / handles | `#815208` | `#E5B16A` |
| Pending proposal surface | `#EEE1C7` | `#463824` |

Hue choice is a design recommendation, not a result of interviews or a claim that
forest is objectively superior to blue. Contrast checks cover representative
foreground/surface pairs, not a complete WCAG audit. Symbols, hatching, and text
carry review states in addition to color. A screening-room alternative changes the
layout to prioritize the candidate and reference comparison, not merely its colors.

## Preview data and honest limitations

The preview uses actual recorded frames from mouse054 and the top 20 suggestions
from the full-frame nearest-example run. Each candidate references the most similar
of the five selected cached example embeddings. Hidden annotations are not used to
label suggestions as correct in the interface.

Frames are snapshots from before, midpoint, and after each candidate. Playback,
frame stepping, search interactions, and timeline editing are local simulations;
they do not drive a real decoder, rerun retrieval, or alter dataset annotations.
Preview state can be saved by the conversation host, but is not application autosave.
The annotation ledger is a local prototype. These distinctions are visible in the UI.

## Research to run next

Recruit 3–5 behavioral researchers, including an experienced BORIS/ELAN user and
someone new to annotation tools. Observe their current workflow before showing the
prototype. Ask them to demonstrate a difficult boundary, an ambiguous event, and how
they recover work after an interruption. Avoid asking only whether they “like” it.

Then give the same five tasks in both layouts: label an event, compare a suggestion
with a reference, adjust its boundary, defer an ambiguous event, and undo a decision.
Measure completion, boundary errors, accidental confirmations, keyboard/mouse
switches, and whether they can explain the difference between a proposal and a label.
Ask them to resume after a short interruption and identify where they left off.

Validate palette preferences on their actual displays and usual room lighting.
Do not assume that a dark appearance is better for every researcher. Ship a theme
choice and configurable shortcuts only after evaluating real workflow needs.

## Representative palette contrast checks

| Pair | Contrast ratio |
|---|---:|
| Light primary text | 10.94:1 |
| Light secondary text | 5.52:1 |
| Light action label | 7.80:1 |
| Light suggestion label | 5.16:1 |
| Dark primary text | 11.32:1 |
| Dark secondary text | 7.25:1 |
| Dark action label | 7.22:1 |
| Dark suggestion label | 5.85:1 |

## Preview verification

An isolated headless Chrome session verified both layouts, accept-and-advance,
defer shortcuts, undo restoring the previous selection and decision, reference
comparison, invalid-boundary validation, ledger visibility, and shortcut suppression
inside text fields. Layout checks found no horizontal overflow at 1024, 736, and
360 px. Desktop and narrow screenshots were inspected. This is prototype QA, not
researcher validation or a complete accessibility audit.
