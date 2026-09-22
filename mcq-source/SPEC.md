# MCQ review-bank spec (Deep Learning midterm, Lessons 1-6)

## Purpose
Review + find weak spots before the real midterm (MCQ, A-D, no calculator, one A4 cheat sheet allowed).
NOT an exam simulation. Every question must teach through its explanation.

## Real-exam facts to align with
- Content: lectures 1-6 + all exercises (homework ex1-3 and in-class notebooks).
- Backprop: NO numeric backprop calculations. Conceptual backprop is fair game (direction of flow, what is cached and why,
  shapes of dW/db/dZ, role of chain rule and g'(z), why zero-init fails, autodiff, unrolling).
- Code questions appear: counting parameters, output shapes, explaining what a line does, fill-in-the-missing-line,
  predicting effect of a callback/hyperparameter. Code questions are welcome in any amount the content supports.
- "Simple calculation" possible, no calculator: every number must be computable by hand in < 1 minute.

## Difficulty (keep it EVEN, never above what was taught)
- Every question must be answerable from the lesson text / the listed notebooks. No outside trivia, no API details the course never showed.
- Mix: ~30% `easy` (direct recall of a definition / term / default / formula meaning),
  ~50% `medium` (understanding: why, compare, predict, 1-2 reasoning steps),
  ~20% `hard` (apply to a new small case, max 3 steps). No trick-on-trick, no double negatives.
- Stem short: readable in ~20 seconds (code snippets excluded).

## Options rules
- Exactly 4 options. Language: Thai prose with English technical terms (as in the lessons).
- Lengths within one question close: longest option <= 1.6x shortest (character count, after stripping HTML) — for short numeric/term options keep them the same form (all numbers, all shapes, etc.).
- Distractor mix: at least one plausible near-miss (sign flipped, term swapped, off-by-one index, forgot bias, wrong default,
  confused train/test behaviour...) AND it's fine to have one that is clearly eliminable if you truly understand.
- Exactly ONE correct option, unambiguous.
- FORBIDDEN: "ถูกทุกข้อ", "ไม่มีข้อใดถูก", "ทั้ง ก และ ข", any reference to letters/positions of other options.
  (Option order is shuffled later by the build script to balance A/B/C/D exactly — so options must be position-independent.)
- Put the correct option at index 0 in the JSON (`answer` is always 0 in writer output; build script shuffles).

## Explanation rules
- `explanation`: HTML (Thai), teach it: why the right answer is right, with the reasoning steps / computation shown.
- `why_wrong`: array of 3 short strings, one per distractor (options[1..3]) — what misconception it represents.
- `ref`: where to re-read, e.g. "Lesson 4 §6.2 Dropout" or "ex3 cell (4)" or "Lecture6 callbacks notebook".
- Math: KaTeX inline `\( ... \)` (escape backslashes properly in JSON: `\\(`). Code in `code` field (plain text, shown in <pre>)
  or inline `<code>` in stem.
- Verify every numeric answer with Python before writing it (run python3). Parameter counts, shapes, EarlyStopping epochs, etc.

## JSON output format (a JSON array, UTF-8, valid JSON)
```json
{
  "id": "L3-07",
  "lesson": 3,
  "topic": "Activation: ReLU vs Leaky ReLU",
  "source": "lesson" | "ex1" | "ex2" | "ex3" | "ex_shallow_nn" | "mnist_notebook" | "callbacks_notebook" | "tensorboard_notebook" | "tuning_notebook",
  "type": "concept" | "code" | "calc",
  "difficulty": "easy" | "medium" | "hard",
  "stem": "HTML string",
  "code": "optional python code string, or omit",
  "options": ["correct", "distractor", "distractor", "distractor"],
  "answer": 0,
  "explanation": "HTML string",
  "why_wrong": ["...", "...", "..."],
  "ref": "Lesson 3 §5B.3"
}
```
