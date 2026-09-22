# Review — q_L3.json (Lesson 3: Cost Function & Backpropagation)

Ground truth read: `0003-cost-function-and-backpropagation.html.txt` (all 8 sections + appendix) and
`/Users/peach/Desktop/Master Degree Prep/deep-learning/Lecture4/ex_shallow_nn.ipynb` (all 28 cells incl. outputs).
All numeric claims re-derived with `python3`/numpy.

**Result: 45 items in → 55 items out. 0 removed, 10 added, 21 edited.**

---

## 1. PASS / FAIL per REVIEW.md criterion

| # | Criterion | Verdict | Notes |
|---|---|---|---|
| 1 | Correctness of the key | **PASS** (0 wrong keys found) | Every number re-verified: `20(2x−5)⁹` @x=3 → 20.0002 numeric slope; `σ'(0)=0.25`; `a(1−a)=0.09` @a=0.9; softmax `[ln3,0,0]→[0.6,0.2,0.2]`; softmax `[1,3,0.5]→[0.1112,0.8214,0.0674]`, `−ln 0.8214=0.1967`, `−ln 0.0674=2.697`; param counts `4-6-6-1 = 30+42+7 = 79`, `3→5→2 = 5×4 & 2×6`, `2→10→1 = 41`; k-fold `1000/5 → 800 train /200 val ×5`; notebook test cell re-run → `Theta1_grad (2,3)`, `Theta2_grad = [−0.29301, −0.25808, −0.25808]`, `len(grad)=9`; `randInitializeWeights(10,1) → (1,11)`, range `[−0.12, 0.12)`; notebook split `600/100/100`, `Training Set Accuracy: 99.5`. |
| 2 | Single correct answer | **FIXED → PASS** | One genuine ambiguity: **L3-23** key read `"…4000 รอบ มากกว่า tanh และ sigmoid"` — "มากกว่า" could attach to *rounds* (false) instead of *accuracy* (true). Rewritten. |
| 3 | Not beyond what was taught | **PASS** | Every stem traces to a slide page (p.1–48) or a notebook cell. No framework defaults the course never showed. |
| 3b | **No numeric backprop items** (lesson-specific rule) | **PASS — no violations found** | No item requires running a backward pass numerically. Nothing from the §3 worked example (p.11–16, δ/Δ/D/θ-update arithmetic) is asked as a computation. Borderline items audited and kept: **L3-09** (`g'=a(1−a)=0.09`) is an activation-derivative lookup from §5A.1/p.25, not a backward pass; **L3-41** asks only for `len(grad)` (a shape count) — the δ numbers appear in the explanation as reading material, never as the task. New **L3-50** is deliberately conceptual (a product with a zero factor), no arithmetic. |
| 4 | Not too hard | **PASS** | Max 3 steps everywhere; no double negatives; no trick-on-trick. Heaviest items (L3-15 param count, L3-26 softmax, L3-46 chain rule) are all ≤3 hand steps. |
| 5 | Not too easy / not trivia | **PASS** | All `easy` items drill a coloured slide definition, a default, or a table cell. No item is answerable by odd-one-out alone. |
| 6 | Difficulty label honest + ~30/50/20 mix | **FIXED → PASS** | Was 29/56/16 with 3 dishonest labels. Now **29 / 53 / 18** (16 easy, 29 medium, 10 hard). |
| 7 | Options (4, position-independent, comparable length, real distractors) | **FIXED → PASS** | Structural rules passed already. The real failure was a **systematic length tell**: the correct option was the *strictly longest* in **18 / 45** items and ranked longest-or-tied in 32/45 — pick-the-longest scored ~70%. Rebalanced to **6 / 55**. No "ถูกทุกข้อ"/"ไม่มีข้อใดถูก"/letter references anywhere. |
| 8 | Explanation teaches + `ref` real | **PASS** | Every `explanation` shows the steps; every `why_wrong[i]` names a misconception aligned to `options[i+1]` (alignment re-checked item-by-item for all 10 new items). All `ref`s resolve to a real §/page or notebook cell. |
| 9 | No duplicates / near-duplicates | **PASS** | No duplicate stems (checked on first 90 plain chars). Closest pair kept — see §4. |
| 10 | Coverage | **FIXED → PASS** | 9 sections/pages had zero items. 10 items added — see §3. |

Invariants preserved: `answer == 0` (55/55), `why_wrong` 3-element (55/55), ids unique (55/55),
valid JSON array, field set unchanged (`id, lesson, topic, source, type, difficulty, stem, [code], options, answer, explanation, why_wrong, ref`).

---

## 2. Table of changes — edits to existing items

| ID | Problem class | What changed |
|---|---|---|
| L3-23 | **Ambiguous key** (crit. 2) + length tell | Key rewritten to `"relu ได้ 100% โดยใช้ 4000 รอบ น้อยกว่า sigmoid และ tanh ที่ใช้ 6000 รอบ"`; distractor 2 sharpened to swap *both* accuracy and rounds; distractor 3's "มากกว่าอีกสองตัวรวมกัน" (arithmetically false, 10000 < 12000) → "มากกว่าอีกสองตัวที่ใช้ 6000"; all 3 `why_wrong` rewritten to match. |
| L3-03 | Length tell | Key trimmed 89→85; distractors 1 & 2 padded. |
| L3-07 | Length tell (key +8 over next) | Key trimmed 86→77; three distractors padded. |
| L3-09 | Length tell | Key trimmed 64→55; distractor 3 padded. |
| L3-12 | Length tell | Key trimmed 81→70; distractors 1 & 3 padded. |
| L3-16 | Length tell (key +11) | Key trimmed 94→82; distractors 2 & 3 padded. |
| L3-17 | Length tell (key +13) | Key trimmed 84→72; all three distractors padded. |
| L3-19 | Length tell | Key trimmed; distractor 3 padded. |
| L3-20 | Length tell **+ grammatical-form tell** | First pass shortened the key to `"output Infinite"` while distractors kept `"output can be Infinite"` — that made the key the odd one out, so it was reverted: key restored to the full slide wording and the three distractors padded to match instead (key 93, longest distractor 95). |
| L3-22 | Length tell | Key trimmed 94→86; distractor 2 padded 72→84; longest is now distractor 1 (88). |
| L3-27 | Length tell (marginal) | Key trimmed 72→70. |
| L3-28 | Length tell (key +9) | Key trimmed 96→87; distractor 2 made the longest (88). |
| L3-30 | Length tell | Key trimmed 71→63; distractors 1–3 padded; key is now the **shortest**. |
| L3-33 | Length tell + ratio 1.49 (worst prose ratio in the bank) | All four rewritten to 50–59 chars; ratio now 1.18. |
| L3-34 | Length tell | Key trimmed 94→89; distractor 3 (the pyramid near-miss) padded to 92 and is now longest. |
| L3-38 | Length tell | Key trimmed 76→66; key is now the shortest. |
| L3-42 | Length tell + **dishonest difficulty** | Key trimmed 70→65; `hard` → **`medium`** (it is a one-step recall that σ(0)=0.5). |
| L3-45 | Length tell | Key trimmed 72→66; distractor 3 padded. |
| L3-05 | **Dishonest difficulty** | `medium` → **`hard`**: picking the correct 4-factor chain out of three near-misses (wrong factor / reversed chain / missing factor) is a genuine 3-step task. |
| L3-41 | **Dishonest difficulty** | `medium` → **`hard`**: requires two shape computations *plus* remembering the bias column (2×3 + 1×3 = 9). |

No item was removed. No `answer` index, `why_wrong` arity, or id was touched.

---

## 3. Items added (coverage gaps)

Nine slide sections and one notebook fill-in had **zero** items before this review.

| New ID | Gap it closes | Source | Diff. |
|---|---|---|---|
| **L3-46** | §1 Chain Rule, p.4–5 — the two worked practice problems the lecturer left unanswered (`y=(2x−5)¹⁰`). The whole proof chapter rests on this and nothing tested it. | lesson | easy / calc |
| **L3-47** | §2.2 p.8 — the definition `L = total number of layers (not counting input layer)`, flagged in the lesson as an exam trap ("ข้อสอบให้รูปเครือข่าย 3 ชั้นแล้วถามค่า L"), plus `δ^(L)=a^[L]−y`. | lesson | medium |
| **L3-48** | §2.1 p.7 — the Andrew-Ng index convention (`θ₁₂^[2]` = *to* node 1, *from* node 2). Load-bearing for every dimension question and untested. | lesson | medium |
| **L3-49** | §2.4 p.10, proof ① — *why* `δ^[L]` has no `g'` factor (the `a(1−a)` cancellation), the single most-emphasised derivation in the chapter. | lesson | **hard** |
| **L3-50** | §3 p.12 conceptual takeaway — a zero outgoing weight zeroes that node's δ *for that round only* (the weight itself still gets a non-zero gradient and revives next round). Written with no arithmetic, to respect the no-numeric-backprop rule. | lesson | **hard** |
| **L3-51** | §5 p.23 — the coloured definition of an activation function (`nonlinear function that transforms or compresses…`), with the slide's own listed trap ("ฟังก์ชันที่บีบค่าให้อยู่ใน [0,1]"). | lesson | easy |
| **L3-52** | §5B.3 p.32 — the **Output Range** row. L3-20 covered only Advantages/Disadvantages; the lesson calls p.32 "หน้าที่ออกสอบบ่อยที่สุดของ §5" and the range row is its most-swapped line. | lesson | easy |
| **L3-53** | §6 p.37 — one-hot encoding (why classes must not stay as 0–3) and the weakness of per-node sigmoid (outputs need not sum to 1). Whole slide page was untested. | lesson | medium |
| **L3-54** | §6 p.38 — why the softmax loss has only the `y_k log h_k` term (`y_k=1` only for the correct class). | lesson | medium |
| **L3-55** | `ex_shallow_nn` — the two remaining fill-in blanks of the forward pass (`a1 = np.concatenate(..., axis=1)` and `z2 = np.dot(a1, Theta2.T)`). L3-38 covered `a0` only. | ex_shallow_nn | medium |

---

## 4. Borderline but kept

- **L3-09 vs L3-36** — both touch `σ'`. Kept: L3-09 applies `a(1−a)` at `a=0.9` inside the δ chain; L3-36 asks for the *maximum* (0.25) from the notebook's `sigmoidGradient`. Different targets, different misconceptions, no shared distractor.
- **L3-14 / L3-15 / L3-35** — three dimension-counting items. Kept: shapes (L3-14), whole-network total across 3 layers (L3-15), and δ-count vs gradient-count in the p.47–48 exercise (L3-35). The lesson leans on `b×(a+1)` in five separate places, so repetition here mirrors the source.
- **L3-05** — distractor 3 is visibly shorter than the other three because it is the "forgot the `g'` factor" option: it legitimately has one fewer term. Raw ratio 1.36, well under the bar, but the shortness is semantically meaningful and cannot be padded without destroying the distractor. Kept as-is.
- **L3-49** — distractor 3 has the largest raw character count (95) purely from two `\(\partial J/\partial a\)` KaTeX spans; rendered, it is the shortest option on screen. No action needed.
- **L3-41** — options `9 / 6 / 7 / 12` give a raw ratio of 2.0, exempt under the "short same-form values (≤28 chars)" clause of both REVIEW.md and `build.py`.
- **L3-20 distractor 2** ("information loss on x<0 (??)") — the slide itself marks this disadvantage with a red `??`. The item asks only what the *table* says, and the explanation flags the lecturer's own doubt, so there is no second defensible key.

## 5. Nothing left unfixable

No item was found that could not be corrected in place. Every failure was either a wording fix, an option-length rebalance, a difficulty relabel, or a coverage gap fillable with a new item.

---

## 6. Re-validation (post-edit)

```
json.load OK                     n = 55
4 options each                   55/55
answer == 0                      55/55
why_wrong 3 elements             55/55
ids unique                       55/55
option-length ratio ≤ 1.6        55/55  (exemptions applied only to all-short same-form sets)
banned phrasings                 0
duplicate option text            0
duplicate stems                  0
correct option strictly longest  6/55   (was 18/45)
difficulty mix                   easy 16 (29%) · medium 29 (53%) · hard 10 (18%)
type mix                         concept 34 · code 12 · calc 9
source mix                       lesson 44 · ex_shallow_nn 11
project build.py                 no L3 errors (only L4-55, L5-52 — other banks, out of scope)
```
