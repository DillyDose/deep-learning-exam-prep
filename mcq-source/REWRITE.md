# Rewrite pass: make every question stand on its own

The real midterm shows only the question. It never says "according to slide 12", never refers to
"the lecturer", "the in-class notebook", "the homework", or "the example we did". Any question that
needs that context to be answerable is broken. Fix every such item.

## Rule 1 — the stem must be self-contained

Delete every reference to the course materials from `stem` and from all four `options`:
slide/page numbers ("ตามสไลด์หน้า 4", "หน้า 27", "p.12"), the lecturer ("ที่อาจารย์ทำสีไว้",
"ตามที่อาจารย์อธิบาย"), the lessons ("ตามบทเรียน"), the notebooks ("ในสคริปต์ที่รันในคาบ",
"notebook", "ex1/ex2/ex3", "การบ้าน", "แบบฝึกหัดในคาบ"), and phrases like "ตามตารางหน้า 4",
"ตามการทดลองในคาบ", "ตัวอย่างในบทเรียน".

Whatever the deleted phrase was carrying must be restated inside the question itself. Examples:
- "ตามสไลด์หน้า 4 อะไรคือนิยามของ deep neural network?" → "ข้อใดคือนิยามของ deep neural network"
- "ลำดับ val_loss ตามตารางหน้า 4 คือ …" → give the sequence in the stem as data of the question.
- "จากโค้ดบนสไลด์หน้า 34 สองบรรทัดสุดท้ายทำอะไร" → put the code in the `code` field and ask about it.
- "ตามนิยามที่สไลด์ทำสีไว้ callback คืออะไร" → "ข้อใดอธิบายบทบาทของ callback ใน Keras ได้ถูกต้อง"

A question may still quote a definition, a formula, a table row or an English phrase from the course —
the course's *content* is exactly what is being tested. What must go is the *pointer to where it was said*.

## Rule 2 — invented scenarios, not the course's own worked examples

If an item's scenario (the numbers, the network shape, the val_loss sequence, the dataset, the code
snippet's exact values) was lifted from a worked example in the lesson or a notebook, build a NEW
scenario instead: different numbers, a different layer size, a different sequence, a different variable
name. The concept and the difficulty stay the same; the specific case must be fresh, so that the item
tests whether the method transfers rather than whether the example was memorised.

Keep it hand-computable without a calculator: small integers, round fractions, powers of 2 or 10,
sigmoid at 0, ReLU, etc. Verify EVERY new number with python3 — the key and the arithmetic behind
each distractor.

Exception: a specific fact that IS the content stays as it is — a formula's constant (He init's 2/n),
a named default (EarlyStopping's `restore_best_weights=False`), a standard architecture the course
actually teaches (784 → 16 → 16 → 10 for MNIST). Do not "freshen" those into something false.

## Rule 3 — what stays untouched

- `id`, `lesson`, `source`, `type` stay as they are. `topic` may be reworded only if the rewrite changed
  what the item asks. `difficulty` must still be honest after the rewrite.
- `answer` stays 0 (correct option first). `why_wrong` stays a 3-element array aligned with options[1..3];
  after any option change, its entry must still describe that option's misconception.
- `ref` STAYS — it is shown to the student after answering, as "where to re-read", and is not part of
  the question. Keep it accurate; if you changed what the item tests, point it at the right section.
- `explanation` may mention the source at the end, but its reasoning must stand alone: show the steps,
  do not justify an answer with "because the slide says so".

## Rule 4 — option quality (unchanged from before)

4 options, exactly one correct, position-independent (no "all of the above", no references to other
options), same grammatical form, comparable length (longest <= 1.6x shortest unless all are short
same-form values), at least one plausible near-miss distractor per item. Do not let the correct option
become the longest one systematically: after your edits, check that the key is strictly longest in no
more than ~25% of the lesson's items.

## Verification before you finish

1. `json.load` parses; item count unchanged (or higher only if you split something); ids unique.
2. `grep` your file for: สไลด์ · หน้า <digit> · p.<digit> · อาจารย์ · lecture · notebook · ในคาบ ·
   ในห้อง · การบ้าน · แบบฝึกหัด · ex1/ex2/ex3 · Exercise — zero hits in `stem` and `options`
   (hits in `ref` are expected and fine; in `explanation` only as a closing pointer).
3. Every numeric answer re-verified in python3.
4. Option-length ratio rule holds; key-longest count <= ~25% of items.
