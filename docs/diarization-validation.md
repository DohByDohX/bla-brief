# Diarization Validation Log

Running log of real-world diarization/transcription quality checks for the
channel-aware diarization feature (v4.8.0+). Each entry documents observed
issues and recommended fixes based on manual review of a recorded meeting's
transcript. Content/topic of meetings is not relevant here — only
transcription and speaker-attribution quality.

---

## 2026-10-01 — Optimus warehouse headcount (Nakul)

**File:** `2026-10-01_1959_Optimus_warehouse_headcount_Nakul.md`
**Quality score:** 6.5/10

### Findings

**Duplicate / split utterances (8+ instances)** — identical or adjacent
speech appearing under two different speaker labels back-to-back:
- "Could we just go back to the thread" → split across `Local` then `Speaker 5`
- "40 hours" → `Local: 40` then `Speaker 5: 40` / `Speaker 5: hours`
- "a bit" → duplicated under `Local` then `Speaker 5`
- "There is another" → duplicated under `Local` then `Speaker 2`

**Micro-segments labeled `Speaker None` (~15 instances)** — short
interjections ("load", "for", "Is", "has", "fabric", "per", "Any", "Uh",
"Hey", "Okay", "Yeah") fail to cluster and fall back to `Speaker None`.
Root cause: VAD minimum segment duration (~0.3s) is too aggressive for
short acknowledgments.

**Word/phrase fragmentation at overlap boundaries** — e.g. "fabric" /
"cover" split across `Speaker None` and `Speaker 0` for what should be one
continuous phrase.

**Possible clustering bleed** — `Local` speaker appearing to double up with
a system-channel speaker on the same utterance ("Could we just go"),
suggesting mic-channel audio may be leaking into system-channel clustering,
or an alignment-offset issue.

### What worked well
- Dual-channel path confirmed active (`Local` label present and used).
- System clustering produced distinct, stable speaker IDs (0–6), not
  collapsed into one cluster.
- Long, single-speaker turns were transcribed and attributed accurately.

### Recommended fixes (priority order)
1. Increase VAD `min_segment` from 0.3s → 0.5–0.7s to reduce `Speaker None`
   fragments from short interjections.
2. Add a post-processing dedup step: merge/flag consecutive lines with
   identical (or near-identical) text under different speaker labels.
3. Audit clustering/alignment to confirm mic-channel (`Local`) embeddings
   are never fed into system-channel clustering; verify
   `resolve_alignment_offset_s()` output on this specific recording.
4. (Optional) Render true overlapping speech with timestamps instead of
   separate sequential lines, to distinguish real overlap from
   mis-segmentation.

---

## 2026-10-02 — Ops Opportunities Next Steps

**File:** `2026-10-02_1430_Ops_Opportunities_Next_Steps.md`
**Quality score:** 4/10 (worse than meeting 1 — much heavier cross-talk)

### Findings

**Pervasive near-duplicate turn boundaries (near 100% of turn transitions)** —
far more severe than the isolated ~8 instances seen in meeting 1. On almost
every speaker change, the trailing clause/fragment of the outgoing speaker's
turn (`Local`) is repeated verbatim as the opening line of the incoming
speaker before that speaker's genuine new content begins, e.g.:
- `Local: decided to like, okay,` → `Speaker 2: decided to like, okay,` → `Local: let's wait for...`
- `Local: No,` → `Speaker 2: No,` → `Local: no, no. But that` → `Speaker 0: no, no. But that`
- `Local: I was confused like` → `Speaker 2: I was confused like` → `Local: why it wasn't working...`

**`Speaker 0` appears to be a pure duplication artifact, not a real speaker** —
every single `Speaker 0:` line in this transcript is an exact duplicate of the
immediately preceding `Local` fragment; it never contributes original text.
By contrast, `Speaker 1` and `Speaker 2` both have substantive, non-duplicate
turns and are correctly separated — those are real distinct speakers.

**Correlation with cross-talk volume** — this meeting had significantly more
real-time back-and-forth (short interjections like "no, no", "yeah", talking
over each other) than meeting 1. Duplication frequency scaled directly with
overlap/cross-talk volume, strongly confirming the overlap-handling bug (not
VAD min-segment threshold) as the dominant root cause.

### What worked well
- `Speaker 1` (Adith) and `Speaker 2` (Colleen) clustering is stable and their
  genuine turns are transcribed accurately and coherently.
- Longer uninterrupted monologue stretches (e.g. Speaker 2's extended
  "TED talk" on prioritization) are clean with no duplication.

### Recommended fixes (priority order — updated from meeting 1)
1. **[Promoted to #1] Fix overlap/turn-boundary duplication.** The trailing
   fragment of an outgoing turn is being re-emitted as the leading fragment of
   the next speaker's turn. Likely cause: the same audio region is picked up
   by both mic-channel VAD and system-channel VAD/clustering, and both get
   rendered without dedup. Needs investigation into segment merge/render logic
   at turn boundaries.
2. Add a dedup/merge post-processing step: collapse consecutive lines
   (regardless of speaker) with near-identical text within a small time delta.
3. Investigate whether the `Speaker 0` cluster in this recording is spurious —
   it may be entirely composed of duplicated `Local` audio bleeding into
   system-channel clustering; consider a confidence/consistency check that
   flags or suppresses clusters that are 100% duplicate text.
4. VAD `min_segment` tuning (0.3s → 0.5–0.7s) — still valid for `Speaker None`
   fragments, but now lower priority relative to the overlap-duplication fix.

---

## 2026-10-05 — Optimus warehouse Stacie numbers discrepancy

**File:** `2026-10-05_1629_optimus-warehouse-stacie-numbers-discrepency.md`
**Quality score:** 6/10 (structured Q&A, fewer overlaps → less duplication than meeting 2)

### Findings

**Moderate duplication at turn boundaries (~30–40% of transitions)** — not as
pervasive as meeting 2, but still systematic. This was a more structured Q&A
format (1-on-1 with fewer interjections), which correlates with fewer duplicate
fragments. Examples:
- `Local: So I'm` → `Speaker 0: So I'm` (duplicate, then continues with original)
- `Local: Yeah,` → `Speaker 0: Yeah,` (duplicate)
- `Local: And then I was also,` → `Speaker 0: And then I was also,` (duplicate)
- `Local: just to get that. Yes.` → `Speaker 0: just to get that. Yes.` (duplicate)

**`Speaker 0` shows mixed artifact + real speaker behavior** — unlike meeting 2
where `Speaker 0` was 100% duplicate, here `Speaker 0` has both duplicate lines
and original substantive content (e.g., "I'm good. Okay, so I just had some
questions..."). This suggests `Speaker 0` may represent a real third participant
(Stacie) but with upstream confusion/overlap on some turns.

**`Speaker 1` introduces new duplication pattern** — in addition to `Speaker 0`
duplicates, this is the first meeting where `Speaker 1` also shows several
duplicate lines ("totally, totally.", "keep it 15%", "you prefer, like, the").
Suggests the overlap bug affects all downstream speaker clusters, not just one.

**`Speaker None` fragments present but less frequent** — single-word/fragment
utterances like "Mm", "There's", "And", "relief", "for", "do", "exactly", "I"
fall to `Speaker None`. Still evidence of VAD threshold being too aggressive, but
less prominent than in meeting 1 (structured speech with fewer short
acknowledgments).

### What worked well
- Long explanatory runs are clean and coherent (e.g., Local's explanation of PIV
  staff and relief calculations).
- Q&A structure with clearer speaker turns (less backchanneling) reduces
  frequency of duplication vs. meeting 2.
- Dominant speakers (`Local`, `Speaker 0`) are mostly distinguishable despite
  duplication noise.

### Recommended fixes (consistent with meeting 2 priority)
1. **Fix overlap/turn-boundary duplication** (primary root cause). This meeting's
   lower duplication rate (~30–40% vs meeting 2's ~100%) strongly confirms the
   pattern: overlap frequency scales with cross-talk volume. A more structured
   meeting style = fewer overlaps = fewer duplication artifacts.
2. Add dedup post-processing to collapse duplicate lines within turn boundaries.
3. Audit whether `Speaker 1` duplication indicates the bug is now affecting
   multiple clusters (suggesting the bug is in the merge/render step, not
   clustering itself).
4. VAD `min_segment` tuning remains valid for `Speaker None` reduction.

---

## 2026-10-06 — Internal Fleet / Optimus Support (Post-Overlap-Fix)

**File:** `2026-10-06_1203.md`
**Quality score:** 8.5/10 (major improvement post-fix)
**Fix tested:** Overlap-duplication bug fix (commit a1d0204, `fix/overlap-duplication`)

### Findings

**Overlap-duplication bug is FIXED** ✅ — No pervasive trailing-fragment
duplication at turn boundaries. Speaker turns flow cleanly with original,
non-repeated content. This meeting had multiple speakers with natural
back-and-forth, and none of the systematic duplication pattern seen in meetings
2 & 3. Examples of clean turns:
- `Speaker 1: ...production day just request for parents...` (original)
- `Speaker 2: good morning team...` (original, not duplicating Speaker 1)
- No `Speaker 0` phantom cluster appearing as pure duplicates

**Remaining `Speaker None` fragments** — still present but fewer and less
intrusive than meeting 1 (5–7 instances: "on", "low", "yeah", "seems", "taking").
These appear to be edge cases (very short segments at overlap boundaries, or
very quiet utterances). VAD `min_segment` tuning would help but is now a
lower-priority cosmetic fix.

**Natural speaker separation maintained** — `Speaker 1`, `Speaker 2`, and
`Local` are consistently and correctly attributed across turns; no cross-cluster
contamination.

### Quality improvement vs previous meetings
| Meeting | Quality | Primary Issue | Status |
|---------|---------|-------|--------|
| 2026-10-02 (Pre-fix) | 4/10 | Overlap duplication ~100% | ❌ Broken |
| 2026-10-05 (Pre-fix) | 6/10 | Overlap duplication ~30–40% | ⚠️ Moderate |
| **2026-10-06 (Post-fix)** | **8.5/10** | **VAD fragments only (~5–7)** | **✅ Fixed** |

### Recommended next steps
1. ✅ **Overlap-duplication fix is effective.** Keep this change. Validate on a
   few more meetings to confirm consistency across varying cross-talk volumes.
2. **VAD `min_segment` tuning (0.3s → 0.5–0.7s)** — now the only
   remaining low-hanging fruit. Would eliminate the remaining `Speaker None`
   fragments and improve end-user experience, though not critical.
3. (Optional) **Speaker None fragment handling** — consider a threshold-based
   merge (collapse very short orphan segments into adjacent speaker turns if
   confidence is below a threshold).

---

<!-- Add new entries above this line as more meetings are validated. -->
