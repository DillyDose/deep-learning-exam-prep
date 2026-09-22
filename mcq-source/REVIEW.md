# Reviewer checklist (MCQ review bank)

Goal of the bank: revision + weak-spot finding before a Deep Learning midterm TODAY.
Real exam: MCQ A-D, no calculator, one A4 sheet, no numeric backprop, content = lectures 1-6 + exercises.

Judge EVERY item against these, and FIX the item in place (edit the JSON) when it fails:

1. **Correctness** — the keyed answer must be true per the lesson/notebook. Re-verify every number with python3
   (param counts, shapes, EarlyStopping epoch, EWMA, softmax, mini-batch counts...). Wrong key = fix the key or the stem.
2. **Single correct answer** — no second defensible option, no ambiguous wording, no two options meaning the same thing.
3. **Not beyond what was taught** — anything not in the lesson text or the listed notebooks must go (or be rewritten to
   what the course actually showed). No obscure framework defaults the course never mentions. No numeric backprop items.
4. **Not too hard** — max 3 reasoning steps, hand-computable in under a minute, stem readable in ~20s, no double negatives,
   no trick-on-trick, no item that needs memorising a long table verbatim.
5. **Not too easy / not pure trivia** — an `easy` item is fine when it drills a definition, a default, or a term that
   matters; but reject items answerable purely by spotting the odd option out without knowing the subject.
6. **Difficulty label honest** — easy/medium/hard set correctly; overall mix roughly 30/50/20 for this lesson.
7. **Options** — exactly 4, position-independent (no "all/none of the above", no references to other options),
   comparable length (longest <= ~1.6x shortest unless all are short same-form values), same grammatical form,
   at least one plausible near-miss distractor; distractors must represent real misconceptions, not nonsense.
8. **Explanation teaches** — states why the key is right with the steps shown, and `why_wrong[i]` names the misconception
   behind each distractor. `ref` points to a real section of the lesson/notebook.
9. **No duplicate / near-duplicate items** inside the lesson — merge or re-aim one of them.
10. **Coverage** — every major section of the lesson and every exercise step listed in the writer's brief has >= 1 item.
    If something is missing, ADD an item.

Invariants you must preserve when editing: `answer` stays 0 (correct option first), `why_wrong` stays a 3-element array
aligned with options[1..3], ids stay unique, file stays a valid JSON array, field set unchanged.
