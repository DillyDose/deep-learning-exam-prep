# Review — `q_L2.json` (Lesson 2: Neural Network Basics)

Reviewed 44 items against every point of `REVIEW.md` + `SPEC.md`, ground truth
`0002-neural-network-basics.html.txt` (Lecture2.pdf p.1–49).
**Result: 44 reviewed → 4 fixed, 6 added → 50 items. Bank passes `build.py` validation with 0 errors, 0 warnings.**

```
Total 50 | type: calc 25 / concept 19 / code 6
difficulty: easy 16 (32%) / medium 25 (50%) / hard 9 (18%)   [target 30/50/20]
answer positions after shuffle: A 13 / B 13 / C 12 / D 12
```

---

## 1. Numeric verification (python3)

Every keyed number was recomputed with numpy/math. **All 44 original keys were numerically correct** — no wrong keys found.

| item | claim | verified |
|---|---|---|
| L2-01 | 5,150 polynomial terms | `2*100 + C(100,2)` = 5150 ✓ |
| L2-03 | 2,352 RGB features | `28*28*3` ✓ |
| L2-06 (old) | 25 params 6→3→1 | `3*7 + 1*4` = 25 ✓ |
| L2-07 | `h = 6x+1` | `3(2x+1)-2` ✓ |
| L2-11 | XOR `[0,1,1,0]` | step-MLP simulated ✓ |
| L2-17 | `g(2)=0.881` | 0.8808 ✓ |
| L2-18 | `g(-10)≈0` | 4.54e-5 ✓ |
| L2-19 | `[0,0,1,0] = x₁∧¬x₂` | ✓ |
| L2-20 | `g(0.5)=0.622` | 0.6225 ✓ |
| L2-23/24 | `(0.6,0.8)`, `√8≈2.83` | ✓ |
| L2-25 | `13` / `[3 4 6]` | ✓ |
| L2-28 | `[[4,2],[4,0]]`, swap → `[[2,4],[3,2]]` | ✓ |
| L2-30 (old) | `(2,3)(3,2)(4,)(4,)` | ✓ |
| L2-31 | `(2,1)` vs `(2,)`, values 5 & 11 | ✓ |
| L2-35 | `θ = [0.7, 0, 1.6]`, J 55.0 → 10.92 | ✓ |
| L2-36 | MSE = 1.2 | ✓ (and 0.8 / 0.8 / 0.24 for the distractor expressions) |
| L2-38 / L2-44 | 13 params; 5×3 + 1×6 = 21 | ✓ |
| L2-40 / L2-43 | h≈1 in both | ✓ |
| new L2-45/46/47/49/50 | `[9,9]`; `[[4,-3],[2,5]]`; OR `[0,1,1,1]`; `[4,7,10]`; XOR `[0,1,1,0]` | ✓ |

---

## 2. Table of changes

| item | problem class | action |
|---|---|---|
| **L2-06** | **Criterion 9 — near-duplicate.** Highest textual similarity in the bank (0.447 with L2-38); L2-06 / L2-38 / L2-44 were *three* items drilling the identical rule `(dest)×(src+1)` with the same "forgot the bias" trap. Also a real coverage hole existed on p.6. | **Re-aimed**: now tests the dense / fully-connected-layer definition and the order *inputs → ×weight+sum → activation* — the lesson's own flagged trap ("สลับลำดับเป็น 'ผ่าน activation ก่อนแล้วค่อยคูณ weight' คือคำตอบลวงยอดฮิต"). `medium calc → easy concept`. Param counting stays covered by L2-38 (plain) and L2-44 (transfer). |
| **L2-30** | **Known issue + Criterion 9.** Confirmed: L2-25 / L2-30 were the top similar code pair (0.344) — identical stem template ("โค้ดต่อไปนี้พิมพ์อะไรออกมาตามลำดับ?"), identical two-`print` shape, both generic-numpy. Worse, the *insight* L2-30 tested (1-D array has no second axis → shape tuple quirk) duplicated L2-31. | **Re-aimed L2-30** (not L2-25, which carries the higher-yield dot-vs-element-wise contrast): now the p.36 slide exercise `A=[[2,3],[5,2],[4,4]] → Aᵀ` by hand, `medium code → medium calc`. The colored fact "no effect on the transpose of a 1 dimensional array" is **preserved verbatim in the explanation**, so no coverage lost. Similarity dropped 0.447 → 0.332 max. |
| **L2-24** | **Criterion 7 — mixed option form.** Options mixed radical and plain decimals (`√8 ≈ 2.83`, `√34 ≈ 5.83`, `8.00`, `4.00`), lens 9/10/4/4 (ratio 2.5, only saved by the short-value exemption). | Normalised to four plain decimals `2.83 / 5.83 / 8.00 / 4.00` (all 4 chars, ratio 1.0). The `√8` derivation moved into the explanation; `why_wrong` now names `√34`, "forgot the square root", and Manhattan distance explicitly. |
| **L2-36** | **Criterion 8 — explanation contradicted the option.** Distractors asserted values their own code does not produce (`np.square(np.sum(...)) ⇒ 1.8` while `why_wrong` said "ได้ 0.8 ไม่ใช่ 1.8"; two others claimed 1.2). Teaching value inverted. | Each option now states the value its expression **really** returns (1.2 / 0.8 / 0.8 / 0.24, all verified). The item is now a clean "which line is MSE" with four truthful computations, and the explanation walks all four. |

## 3. Items added (coverage gaps, Criterion 10)

| id | gap it fills | diff / type |
|---|---|---|
| **L2-45** | §5A p.22–23 vector-operation properties (commutative / associative / **distribution**) had **zero** items, although the lesson calls it out ("จุดนี้ออกสอบแน่นอน" — don't carry commutativity over to matrix multiply). `3([1,2]+[2,1]) = [9,9]`. | easy / calc |
| **L2-46** | §5B p.29–30 matrix addition + scalar multiplication had **zero** items (only appeared as a distractor in L2-29). Uses the slide's own blank: `[[1,2],[2,4]] + [[3,-5],[0,1]]`. Distractors = sign slip, multiplied instead of added, subtracted. | easy / calc |
| **L2-47** | §4 p.16–17: the lesson names the AND↔OR bias swap as "ตัวเลือกลวงที่พบบ่อยที่สุด", yet no item tested it directly (L2-39 only listed θ-sets). `θ₀: −30 → −10` ⇒ OR. | medium / calc |
| **L2-48** | §6 p.42 `computeCost` for-loop block had **no** item (only the vectorized version, L2-34, was covered). Tests `X.shape[0]` / `X.shape[1]` = the "rows first, columns second" rule from p.27 applied to code. | easy / code |
| **L2-49** | §6 p.41 worked example (`h=1+2x₁+3x₂`, 3 rows → `[4,7,10]`) was uncovered; L2-32 only did the single-example case. Distractors: forgot the `x₀` column, double-counted bias, swapped θ₁/θ₂. | medium / calc |
| **L2-50** | §7 mechanism box / §2 p.10: XOR-by-composition was only ever *asserted*. Applies the §7 pattern to a new gate set (two one-sided gates + OR) — genuine transfer, 3 steps. | hard / calc |

Coverage after additions: **every numbered section §1 … §8 (plus §1A, §3A, §5A, §5B) has ≥ 2 items**; every slide block with a worked calculation (p.2, 3, 6, 10, 13, 16–18, 19, 22–26, 27–37, 38–44, 45–47, 48–49) is represented.

---

## 4. Borderline but kept

| items | why it was kept |
|---|---|
| **L2-38 / L2-44** (param counting, 2→3→1 and 2→5→1) | Same rule, but L2-44 is an *extension* item (matrix shapes + recount for a changed architecture) that the lesson's own "เปลี่ยนแล้วเกิดอะไร" box sets up. Two levels of one skill, not a duplicate. Similarity after the L2-06 re-aim is well below threshold. |
| **L2-40 / L2-43** (sim 0.308, both sigmoid forward passes) | L2-40 verifies the *memorised* XNOR network (recall + check); L2-43 is a **new** network the student has never seen (transfer). Different cognitive task, different numbers, different output gate. |
| **L2-32 / L2-49** (sim 0.332, both `Xθ`) | L2-32 = one example, scalar result, tests "don't forget `x₀`". L2-49 = m examples, column result, tests building `X` and the dimension check `(3×3)(3×1)`. The lesson teaches them as two separate slides (p.39 vs p.41). |
| **L2-19 / L2-50** | L2-19 identifies a single neuron's gate; L2-50 uses that same gate type as a *building block* of a two-layer network. Deliberate progression, explicitly cross-referenced in L2-50's explanation. |
| **L2-03, L2-26, L2-38** option-length ratios > 1.6 | Exempt: all options are short same-form values (≤ 5 chars: bare numbers). Confirmed by `build.py`'s own exemption rule. |
| **L2-36 labelled `hard`** | Still requires computing `h−y`, squaring, summing and dividing on a vector the student has not seen (`h₅=6`), plus discriminating four numpy expressions. Kept `hard`; it is the upper end of medium at worst. |

---

## 5. PASS / FAIL per criterion

| # | Criterion | Verdict |
|---|---|---|
| 1 | **Correctness** — key true per lesson | **PASS**. All 50 keys recomputed in python3 (matrix products, gate truth tables, param counts, GD step, MSE/MAE, shapes). No wrong key found in the original 44; the only value-level defect was L2-36's *distractor* labels, now fixed. |
| 2 | **Single correct answer** | **PASS**. No second defensible option after the L2-36 fix (previously three options claimed the same value). No two options mean the same thing (checked by exact-match dedupe after HTML/LaTeX stripping). |
| 3 | **Not beyond what was taught** | **PASS**. `source` is `lesson` for all 50; every item traces to a specific page of Lecture2.pdf p.1–49. No numeric backprop, no framework defaults the deck never showed. L2-50 is an application of §7's own compositionality argument, not outside material. |
| 4 | **Not too hard** | **PASS**. Longest chain is 3 steps (L2-50: two hidden gates → OR; L2-35: predict → gradient → update). All sigmoid values are the memorised saturation set `g(±10), g(±30), g(0), g(0.5), g(2)`. No double negatives, no trick-on-trick. |
| 5 | **Not too easy / not trivia** | **PASS**. Every `easy` item drills a definition or default that matters (dense-layer order, bias unit, MLP definition, `shape[0]/shape[1]`, matrix addition). No item is answerable by odd-one-out spotting. |
| 6 | **Difficulty labels honest** | **PASS**. 32 / 50 / 18 vs the 30/50/20 target. Re-labelled L2-06 (`medium calc` → `easy concept`, it is now pure definition recall) and L2-30 (`code` → `calc`, it is now hand computation). |
| 7 | **Options** | **PASS**. Exactly 4 everywhere; position-independent (no "ถูกทุกข้อ" / "ไม่มีข้อใดถูก" / letter references — checked against `build.py`'s BANNED list); length ratio ≤ 1.6 after stripping HTML except three short-value items that qualify for the exemption; each item has ≥ 1 near-miss distractor (sign flip, missing bias column, swapped i/j, swapped sum/square, off-by-one in θ index). |
| 8 | **Explanation teaches** | **PASS**. Every explanation shows the computation, not just the verdict; all 50 `why_wrong` arrays have exactly 3 entries naming a *misconception* rather than "wrong". `ref` strings all resolve to a real `§` heading + real Lecture2.pdf page range (verified against the lesson's §1 … §8 and the p.1–49 appendix). |
| 9 | **No duplicates / near-duplicates** | **PASS** (after fixes). Max pairwise stem+options similarity fell from **0.447 → 0.332**; `build.py` reports 0 near-duplicate-stem warnings. Flagged pair L2-25/L2-30 resolved by re-aiming L2-30; the larger triple L2-06/38/44 resolved by re-aiming L2-06. |
| 10 | **Coverage** | **PASS** (after 6 additions). Previously uncovered: vector properties (p.22–23), matrix add / scalar multiply (p.29–30), AND↔OR bias swap (p.16–17), for-loop `computeCost` (p.42), the p.41 multi-example exercise, XOR by composition. All now have an item. |

---

## 6. Out of scope, worth flagging

`build.py` over the whole bank currently **fails** on two items outside Lesson 2:

```
L4-55: duplicate options
L5-52: duplicate options
```

Lesson 2 alone builds clean (verified by running `build.py` with the lesson-count guard restricted to L2: 0 errors, 0 warnings). The full `dl-mcq-review.html` cannot be regenerated until L4-55 and L5-52 are fixed by their reviewers. I did **not** touch `dl-mcq-review.html`.
