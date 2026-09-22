# Review — q_L4.json (Lesson 4: Deep Networks & Optimizers)

**Before:** 52 items · easy/medium/hard = 13/34/5 (25/65/10 %)
**After:** 61 items · easy/medium/hard = 18/31/12 (30/51/20 %) — matches the 30/50/20 target
Sources: 49 lesson / 12 ex2 · Types: 34 concept / 14 code / 13 calc
Ground truth read: `0004-deep-networks-and-optimizers.html.txt` (§1–§7A + appendix) and `ex2_6880318026.ipynb` (all 50 cells).

---

## 1. Changes made

| id | class of problem | what was wrong | fix |
|---|---|---|---|
| L4-19 | **rendering bug** | `why_wrong[0]` was double-escaped: `\\\\(\\\\alpha\\\\)` → renders as literal backslashes instead of KaTeX | rewrote as `\\(\\alpha\\)` |
| L4-47 | **rendering bug** | `explanation` contained two raw `<` characters (`\(0.12<0.6\)`, `\(0.59<0.6\)`) inside an HTML string — the browser eats everything up to the next `>`, destroying steps 1–2 of the worked answer | escaped to `&lt;` |
| L4-11 | **broken why_wrong text** | `why_wrong[0]` read “เอา 6,400 หารด้วย … ผิดขั้น หรือคูณ 64 กับ 20 …” — an unfinished sentence with a literal ellipsis | rewritten to name the actual misconception for option 1 (1,280 = 64 × 20) |
| L4-33 | **why_wrong ↔ option mismatch** | option 2 is `(1.2, 1.6)` = w/r, but `why_wrong[1]` described dividing by ‖w‖ (which gives `(0.6, 0.8)`) — the stated misconception did not produce the stated option | rewritten: “หารด้วย \(r\) แทนที่จะคูณด้วย \(r/\|w\|_2\) — ได้เวกเตอร์ที่ยาว 2 ไม่ใช่ 5” (verified ‖(1.2,1.6)‖ = 2) |
| L4-36 | **near-duplicate** of L4-47 (both = “apply mask, divide by keep_prob, pick the vector”) | re-aimed to the compensation **multiplier** table (0.9→×1.111, 0.8→×1.25, 0.5→×2) plus the slide’s own “+10 % is really +11.1 %” gotcha, which no item covered | new stem/options/explanation/why_wrong |
| L4-37 | **near-duplicate** of L4-45 (both = “backward reuses the same mask and the same keep_prob”, identical distractor logic) | re-aimed to the *consequence*: a dropped unit gets δ = 0 so its weights are not updated that round (p.31 / lesson §6.2 worked example) | new stem/options/explanation/why_wrong |
| L4-20, L4-24, L4-39 | **difficulty label** | labelled `medium` but each is direct recall of a single slide pair | → `easy` |
| L4-21, L4-33, L4-44, L4-50, L4-51 | **difficulty label** | labelled `medium` but each needs 3 chained steps (decay-window indexing, norm→factor→rescale, shape trace through two matmuls, slice+where-g′-is-evaluated, sigmoid(0)=0.5 ⇒ contour level 0) | → `hard` |

### Items added (coverage gaps)

| id | gap it closes | diff |
|---|---|---|
| L4-53 | §2 / p.6 — “faster **and** keeps `correct trajectory` / reliable convergence” was untested (the lesson flags dropping the second condition as a classic MCQ trap) | easy |
| L4-54 | §3.4 / p.11 — **Shuffling** before splitting mini-batches: no item existed | medium |
| L4-55 | §4.1 / p.15 — recursive expansion of EWA; coefficient of `value_{t-2}` = β²(1−β). This is literally the lesson’s own quiz 6 and had no bank item | hard |
| L4-56 | §6.1 / p.27 — Keras code `kernel_regularizer` vs `kernel_constraint=max_norm(1.)`, incl. the sign-flip trap (λ up = stronger, **r down** = stronger). No code item existed for §6.1 | medium |
| L4-57 | §6.2 / p.31 — parameter count of the Dropout `Sequential` model (235,500 + 30,100 + 1,010 = **266,610**; Flatten/Dropout = 0 params). The lesson computes this explicitly; no item used it | hard |
| L4-58 | **ex2 fill-in step** “Add regularization term (1 line)” had **zero** items — only the two dropout fill-ins were covered. Tests why `temp1[:, 1:]` (drop the bias column) | medium |
| L4-59 | §7 / p.34 — augmentation must preserve the label (6 → 9 under 180° rotation). The lesson’s mechanism box makes this the key takeaway | easy |
| L4-60 | ex2 cells (20)–(21) + (30)–(32) — the 250-row split into 150/50/50 and the **baseline** 86 / 80 / 82 % were untested | medium |
| L4-61 | §4.4 / p.18 — the four labelled loss curves (`way too high: diverges`, `too small: slow`, `too high: suboptimal`, `just right`) + “Start high then reduce”. The lesson calls this “ชุดคำตอบสำเร็จรูปของข้อสอบ”; no item existed | medium |

**Items removed:** none. **ids reused:** none (new ids L4-53 … L4-61).

---

## 2. Numbers re-verified with python3

All of these were recomputed (`nums.py`) and matched the keyed answers **and** the stated distractor arithmetic:

- params `[3,5,4,1]` = 49 (distractors 39 = no bias, 51 = bias on wrong axis) · ∂J/∂θ⁽²⁾ shape 4×6
- iterations: 6400/64 × 20 = **2,000** (distractor 1,280 = 64×20) · ⌈1000/64⌉ = **16**, last batch **40**
- EWMA β=0.9: v₂(2,4) = **0.58** (distractor 0.60 = forgot β·v₁) · momentum v₄: +1×4 → **0.3439**, ±1 alternating → **−0.0181** · θ₂ = **0.971** · v₅ after g=0 → **0.3095**
- LR decay from 0.2: linear halve/5 at epoch 12 → **0.05**; exponential ×0.1/8 at epoch 12 → 0.02
- bias correction t=1: 1/(1−0.9) = **10**, 1/(1−0.999) = **1000**; with g=0.2 → v̂=0.2, ŝ=0.04
- max-norm (6,8), r=5: ‖w‖=10, factor 0.5 → **(3,4)**; the (1.2,1.6) distractor has norm 2
- dropout: [3,6,1.5,9]∘[1,0,1,1]/0.75 = **[4,0,2,12]**; [0.3,1.2,0.9,0.6] with rand [0.12,0.65,0.59,0.93], kp=0.6 → mask [1,0,1,0] → **[0.5,0,1.5,0]**; inverted-threshold distractor → [0,2,0,1]; ×kp distractor → [0.18,0,0.54,0]; multipliers 1/0.9=1.111, 1/0.8=1.25, 1/0.5=2
- early stopping 0.50/0.41/0.38/0.39/0.37/0.40/0.42, patience 2 → **stop end of epoch 7, restore epoch 5**
- EWA coefficients β=0.9: β²(1−β)=0.081, β(1−β)=0.09, total after t=2 = 1−β²=0.19
- Keras Dropout model: **266,610** (distractors 266,200 no-bias, 235,500 first layer only, 315,010 using 784 as the 2nd Dense input)
- ex2: mask shape (150,10); λ argmin(val) = **0.03** (0.262478); keep_prob argmin(val) = **0.9** (0.243792) ⇒ rate 0.1; split 150/50/50 of 250; baseline 86/80/82, gap 6 pts; test acc 82 → 88 (L2) / 86 (dropout)
- lesson figures quoted as-is and checked against source text: 261 params for [2,10,10,10,1], 81.714286 / 99.142857 / 99.857143 %, 15.7× gradient shrink

---

## 3. PASS/FAIL per REVIEW.md criterion

| # | criterion | verdict |
|---|---|---|
| 1 | Correctness of the keyed answer | **PASS** after fixes — every numeric key recomputed in python3; every conceptual key traced to a quoted slide phrase |
| 2 | Single correct answer / no ambiguity | **PASS** — no second defensible option found; near-misses always falsified by an explicit clause (e.g. L4-42 opt 2 shares the true first half but adds “0.5 ทุกชั้นเสมอ”) |
| 3 | Not beyond what was taught | **PASS** — everything traces to Lecture 4 pp. 2–35 or ex2. No numeric backprop items (L4-03/05/37/50 are shape/role/consequence only, which SPEC explicitly allows) |
| 4 | Not too hard (≤3 steps, <1 min by hand) | **PASS** — hardest are 3-step (L4-40 patience walk, L4-57 three products, L4-55 two substitutions). No double negatives; the only negative stem is L4-31 (“ข้อใดไม่ใช่”), which is single-negative and the format the lesson itself predicts |
| 5 | Not trivia / not answerable by odd-one-out | **PASS** — all 18 easy items drill a named definition, default, or slide-quoted pair |
| 6 | Difficulty labels honest + 30/50/20 mix | **PASS** after relabelling 8 items — now 30/51/20 |
| 7 | Options: 4, position-independent, comparable length, near-miss distractor | **PASS** — validator enforces 4 options, no “ถูกทุกข้อ/ไม่มีข้อใดถูก/ทั้ง ก และ ข”, no references to other options; worst length ratio after HTML-strip is **1.54** (L4-55, and its options are 13–20 chars i.e. short same-form anyway); worst among long-option items is **1.29** |
| 8 | Explanation teaches + why_wrong names the misconception + real `ref` | **PASS** after fixing L4-11, L4-19, L4-33, L4-47. Every `ref` points at a real §/slide page or a real ex2 cell |
| 9 | No duplicates | **PASS** after re-aiming L4-36 and L4-37 (the two duplicate pairs found) |
| 10 | Coverage of every lesson section + every ex2 step | **PASS** after adding 9 items. Section map below |

### Coverage map (every section has ≥1 item)

§1 → 01 · §1.1 → 02, 04 · §1.2 → 03, 05 · §2 → 06, 07, **53** · §3.1 → 08 · §3.2 → 09 · §3.3 → 10 · §3.4 → 11, 12, **54** · §3.5 → 13 · §3.6 → 14 · §4.1 → 15, 16, **55** · §4.2 → 17, 18, 19 · §4.3 → 20 · §4.4 → 21, 22, **61** · §5 → 23 · §5.1 → 24 · §5.2 → 25 · §5.3 → 26, 27 · §5.4 → 28 · §5.5 → 29, 30 · §6 → 31 · §6.1 → 32, 33, **56** · §6.2 → 34, 35, 36, 37, 38, **57** · §6.3 → 39, 40 · §7 → 41, **59** · §7A → 42

ex2 steps: helper funcs/relu → 50 · data load + split + baseline → **60** · dropout fill-in fwd (4 lines) → 43, 44, 47 · reg_term fill-in (1 line) → **58** · dropout fill-in bwd (2 lines) → 45 · optimize1 / keep_prob default → 46 · decision boundary → 51 · validationCurve λ → 48 · validationCurve dropout → 49 · comparison / conclusion → 52

---

## 4. Borderline items kept (with reasoning)

- **L4-16** (β 0.9 → 0.98 ⇒ memory ~10 → ~50 values). The `1/(1−β)` rule is not printed on a slide, but the lesson derives the equivalent fact (table of `1−βᵗ`: at β=0.9, t=10 → 0.6513, i.e. “~10 steps to reach 65 %”) and the §4.4 mechanism box contrasts β=0.5 “จำสั้น” vs β=0.9 “จำยาว”. The key is falsifiable from the lesson alone. **Kept.**
- **L4-31** negative stem (“ข้อใด **ไม่ใช่** …”). Single negative, and the lesson explicitly predicts this exam format for the p.26 list. **Kept.**
- **L4-48 / L4-49** are structurally parallel (read a validation curve, pick the argmin). They are *not* duplicates: different tables, different answers, and the ex2 brief requires both steps. L4-49 additionally tests the keep_prob ↔ dropout-rate inversion. **Both kept.**
- **L4-43 / L4-47** both touch the 4-line forward dropout block. L4-43 asks what one *line* does; L4-47 requires executing all four on real numbers including the threshold. Distinct skills. **Both kept.**
- **L4-51** (contour level 0) leans on `np.round(sigmoid(z2))` ⇔ `z₂ ≥ 0`. Two inferences, promoted to `hard`. Fully contained in the ex2 cells. **Kept.**
- **L4-58** explanation notes the notebook’s own quirk (the live `J = …` line does **not** add `reg_term`, only the gradient carries `λ/m·Θ`). The item is not keyed on that quirk — it is keyed on the `[:, 1:]` bias slice — so the quirk is mentioned as a footnote only. **Kept.**

## 5. Unfixable / out of scope

Nothing. All findings were fixable in place; the file remains a valid JSON array with `answer: 0`, 3-element `why_wrong`, unique ids and an unchanged field set (`code` optional, as before).

---

## 6. Length-tell rebalance (follow-up pass)

**Problem.** Measuring option lengths with HTML stripped (`re.sub(r"<[^>]+>","",s)` then whitespace-collapsed), the key (`options[0]`) was **strictly the longest of the four in 27 of 61 items** — chance is ~15 — and **longest-or-tied in 41 of 61**. A student could score far above chance by always picking the longest option. Other lessons in this bank sit near 25 %; L4 was the outlier.

**Result.**

| metric | before | after | budget |
|---|---|---|---|
| key strictly longest | 27 / 61 (44 %) | **11 / 61 (18 %)** | ≤ 16 |
| key longest-or-tied | 41 / 61 (67 %) | **25 / 61 (41 %)** | ≤ 34 |
| key strictly shortest | 2 / 61 | **2 / 61** (unchanged) | — |
| worst intra-item length ratio | 1.54 (L4-55) | **1.54 (L4-55, untouched)** | ≤ 1.6 |

**Method.** 16 items rebalanced. In 15 of them the key was left exactly as written and one or two *distractors* were extended with equally specific, same-register wording, so the key falls to rank 2 or 3 by length instead of rank 1. Only L4-07 had its key touched, and only by deleting a single filler word (`ที่`) so the key and its mirror-image distractor become the same length — no content was removed anywhere. No key was made the shortest or the odd one out in form; all four options in every touched item keep the same grammatical shape.

| id | what changed | lengths before → after (key first) |
|---|---|---|
| L4-01 | padded distractors 2 and 4 (`…จากโมเดลเพียงตัวเดียวโดยไม่ต้องเทรนซ้ำ`, `…ช่วยคำนวณทุกครั้งที่เทรนโมเดล`) | [90, 81, 87, 80] → [90, **107**, 87, **84**] |
| L4-05 | padded the transpose distractor (`…ก่อนนำไปอัปเดต weight ในแต่ละรอบ`) | [93, 82, 86, 83] → [93, 82, **97**, 83] |
| L4-07 | key: dropped filler `ที่` so it matches its swapped mirror exactly; padded distractors 3 and 4 | [97, 94, 87, 92] → [**94**, 94, **98**, **100**] |
| L4-09 | padded the “always a serious problem” distractor (`…ที่เทรนโมเดลใหม่เสมอ`) | [121, 109, 108, 106] → [121, **123**, 108, 106] |
| L4-10 | padded the “more accurate direction” distractor (`…เมื่อชุดข้อมูลเทรนมีขนาดใหญ่ขึ้น`) | [120, 106, 105, 104] → [120, **124**, 105, 104] |
| L4-14 | padded the “lr too high” distractor (`…ไม่เช่นนั้น cost จะไม่ลดลง`) | [99, 84, 77, 78] → [99, **103**, 77, 78] |
| L4-16 | padded the reversed-β distractor (`…เส้นจึงตอบสนองไวขึ้นแต่ noisy ขึ้นมาก`) | [99, 95, 95, 94] → [99, **105**, 95, 94] |
| L4-20 | padded the “auto-raises α” distractor (`…จึงก้าวต่อไปได้`) | [99, 93, 86, 85] → [99, **109**, 86, 85] |
| L4-22 | padded the `decay=` distractor (`…รับได้เฉพาะตัวเลขคงที่เท่านั้น`) | [108, 105, 93, 98] → [108, **115**, 93, 98] |
| L4-23 | padded the role-swap distractor (`…ตามขนาดของ gradient ล่าสุดในแต่ละชั้น`) | [105, 100, 100, 86] → [105, **114**, 100, 86] |
| L4-24 | padded distractors 2 and 4 (`…ใหญ่ต่อเนื่อง…ในทุก step`, `…ไม่เสถียรจนล้มเหลว`) | [106, 98, 89, 82] → [106, **117**, 89, **91**] |
| L4-27 | padded the missing-exponent distractor (`…ในทุกก้าว`) | [118, 114, 106, 98] → [118, **124**, 106, 98] |
| L4-29 | padded the “drops adaptive scaling” distractor (`…ของการเทรนโมเดลทั้งหมด`) | [118, 100, 94, 99] → [118, **123**, 94, 99] |
| L4-42 | padded the all-ReLU distractor (`…คือสัญญาณของโมเดลที่ดีที่สุดเสมอ`) | [99, 90, 89, 88] → [99, **103**, 89, 88] |
| L4-48 | padded the train-error distractor (`…จน underfit เช่นเดียวกัน`) | [128, 120, 114, 112] → [128, **133**, 114, 112] |
| L4-59 | padded the rotate-15° distractor (`…กับงานจำแนกตัวเลขทุกกรณีเสมอ`) | [100, 97, 96, 93] → [100, **106**, 96, 93] |

**Invariants re-verified programmatically after the pass:** `json.load` parses; 61 items; every item has 4 options, `answer: 0` and a 3-element `why_wrong`; the id list, field set, `stem`, `explanation`, `why_wrong`, `ref`, `topic` and `difficulty` of every item are byte-identical to the pre-pass file; no intra-item length ratio exceeds 1.6 (worst is still L4-55 at 1.54, whose four options are 13–20 chars of bare formula). Because only distractor *wording* was extended — never its claim — each padded option still says exactly what its `why_wrong` entry says it says; no `why_wrong` or `explanation` needed editing.
