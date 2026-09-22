# Review — `q_L5.json` (Lesson 5: vanishing/exploding gradients, init, BN/LN, Keras, IMDB ex3)

**Before:** 51 items · **After:** 59 items (0 removed, 8 added, 3 edited in place)

Ground truth used: `0005-vanishing-gradients-and-initialization.html.txt` (all 7 sections + appendix),
`Lecture5/deep_lecture5_mnist_69.ipynb` (cells 5–30), `ex3_6880318026.ipynb` (cells 4–38).
Every numeric key was recomputed with `python3` (see "Numbers verified" below).

---

## 1. Table of changes

| id | class of problem | what was wrong | fix |
|---|---|---|---|
| L5-36 | distractor rationale wrong | option `40` was justified as "เอา 40% ของ 100 มาใช้ตรงๆ ไม่เกี่ยวกับ batch" — i.e. labelled as nonsense. It is actually a *real* misconception: `⌈25,000×0.4/256⌉ = ⌈10,000/256⌉ = 40` = the number of **validation** batches. | rewrote `why_wrong[2]` to name the real misconception; added one clause to `explanation` so the item teaches it |
| L5-37 | distractor contained false arithmetic | option 2 claimed "ใช้ batch_size = 256 … 25,000/256 ≈ 782" (it is ≈ 98) — a self-refuting option, eliminable without understanding | replaced with "evaluate ใช้ batch_size เท่าตอนเทรน (256) ไม่มีค่า default ของตัวเอง" (real misconception: thinking `evaluate` inherits the training batch size); option 3 tightened; `why_wrong[1..2]` rewritten |
| L5-33 | metadata | `topic` said "Sequential สองแบบ + None ใน summary" but the stem only asks about `None` | `topic` → "None ใน model.summary()" |
| **L5-52** *(new)* | coverage gap §5.3 | no item on `(X, y), (Xtest, ytest) = mnist.load_data()` — an unpacking form the lesson flags explicitly as a fill-in-the-code trap | added (easy/code) |
| **L5-53** *(new)* | coverage gap §5.1/§5.12 | `evaluate` return order `loss, acc` is flagged in the lesson as "ลำดับนี้ออกสอบบ่อย" but appeared only inside other items' explanations | added (easy/code) |
| **L5-54** *(new)* | coverage gap §5.0 p.20 | Keras identity (high-level API, Chollet, TF/PyTorch/JAX) is coloured slide text with a named trap ("Keras เป็น backend เอง") and had no item | added (easy/concept) |
| **L5-55** *(new)* | coverage gap §5.10 | the lecture's flagship calculation chain 60,000 → 48,000 → 750 → ×40 = 30,000 updates had no item (L5-36 only covers the IMDB analogue) | added (hard/calc) |
| **L5-56** *(new)* | coverage gap §3.0B p.10–11 | slides 10–11 (Linear vs Saturating zone — *why* BN targets mean 0/var 1) were entirely unrepresented | added (medium/concept) |
| **L5-57** *(new)* | coverage gap §1.2 p.3 | dying ReLU + Leaky ReLU appeared only as a distractor rationale elsewhere; it is coloured slide text | added (medium/concept) |
| **L5-58** *(new)* | coverage gap §3.2 p.14 | coloured text "add a BN layer as first layer ⇒ not need to standardize your training set" had no item (only mentioned inside L5-19's explanation) | added (medium/concept) |
| **L5-59** *(new)* | coverage gap §4 mechanism box | the batch-size-1 case (BN collapses to β, LN still works) — the numeric reason for "Independent of batch size" — had no item | added (hard/calc) |

---

## 2. Numbers verified with `python3`

| claim | item(s) | verified |
|---|---|---|
| MNIST 784→16→16→10 = 16·785 + 16·17 + 10·17 = **13,002**; forgot-bias variant = **12,960** | L5-34 | ✓ |
| BN `4n` split: non-trainable 2(16+16+10) = **84**; total 13,002+64+64+40 = **13,170**; trainable **13,086** | L5-18 | ✓ |
| BN at input: 4·784 = **3,136**, trainable **1,568**; model total **16,266**, non-trainable **1,632** | L5-19, L5-58 | ✓ |
| IMDB 10000→16→16→1 = 16·10001 + 272 + 17 = **160,305**; distractors 160,272 (no bias), 160,016 (layer 1 only), 160,322 (2-node softmax) | L5-35 | ✓ all four |
| IMDB batches/epoch = ⌈25,000·0.6/256⌉ = **59** (log `59/59`); retrain ⌈25,000/256⌉ = **98** (log `98/98`); validation ⌈10,000/256⌉ = **40** | L5-36 | ✓ |
| `evaluate` bars: ⌈25,000/32⌉ = **782**, ⌈10,000/32⌉ = **313** | L5-37 | ✓ |
| MNIST: 48,000/64 = **750** batches, ×40 = **30,000** updates; no-split variant ⌈60,000/64⌉·40 = **37,520** | L5-55 | ✓ |
| Xavier 784→16: 2/800 = **0.0025**; He 2/784 = **0.00255**; 1/800 = 0.00125; 2/16 = 0.125 | L5-11 | ✓ |
| Var(z) with 400 unit inputs = 400 ⇒ σ = **20** | L5-09 | ✓ |
| clip-by-norm [6,−8], c=5: ‖g‖=10 ⇒ **[3,−4]** | L5-06 | ✓ |
| BN of z=[1,3,1,3], γ=2, β=1: μ=2, σ²=1 ⇒ z̃(z=3) = **3**; distractors 2 / 4 / 7 each match their stated error | L5-16 | ✓ |
| moving mean 0.9(10)+0.1(20) = **11** | L5-22 | ✓ |
| LayerNorm μ⁽⁴⁾ = 903/3 = **301**; 903/4 = 225.75 (wrong divisor) | L5-27 | ✓ |
| ex3 `np.argmin(val_loss)+1` = **2** (val_loss[1]=0.2895); MNIST notebook = **13** (0.1642) | L5-40 | ✓ against notebook output |
| ex3 train loss 0.50 → **0.0025** (epoch 20), val_loss **0.7689**, min **0.2895** @ epoch 2 | L5-39 | ✓ against notebook output |
| ex3 L2 prediction 0.3036 ⇒ class 0, actual 0 | L5-48 | ✓ against notebook output |
| Dropout rate 0.6 ⇒ keep 0.4, compensation 1/0.4 = **2.5**; rate 0.2 ⇒ 1.25 | L5-50, L5-51 | ✓ |
| slide-38 model params 235,500 + 30,100 + 1,010 = 266,610 | L5-51 expl. | ✓ |

## 3. Keras API audit (criterion 3 — "nothing the course never showed")

Every API detail asserted in the bank was traced to the lesson text or a listed notebook:

| API detail | asserted in | shown by course at |
|---|---|---|
| `Dense` default `kernel_initializer='glorot_uniform'` | L5-30 | §2.3 and §5.1 line-note (stated twice, verbatim) |
| `evaluate` default `batch_size=32` | L5-37 | §5.12 ("batch_size ของ evaluate ค่า default = 32") |
| `fit` default `batch_size=32` | L5-36 expl. | §5.10 line-note |
| BN `momentum` as the EMA hyper-parameter; Keras default 0.99 | L5-21, L5-22 | §3.3 + its calculation box |
| `SGD(clipvalue=…)` / `SGD(clipnorm=…)` | L5-06 | §1.2 closing line |
| `Input(shape=…)` vs `input_shape=` | L5-45, L5-47 | §6.1 coloured trap |
| `kernel_regularizer=regularizers.l2(λ)`, kernel = weights only | L5-49 | §7.2 |
| `Dropout` disabled at evaluate/predict, Param # = 0 | L5-50, L5-51 | §7.2 "ผลลัพธ์ที่คาด" |
| `to_categorical` ↔ `categorical_crossentropy`, integer ↔ `sparse_…` | L5-32, L5-43, L5-44 | §5.6, §6 table |
| `train_test_split` from `sklearn.model_selection`, returns 4 blocks; `validation_split` cuts the tail without shuffling | L5-46 | §6.2 |
| Keras 3 backends TF/PyTorch/JAX | L5-54 (new) | §5.0 p.20 |
| `mnist.load_data()` nested-tuple return | L5-52 (new) | §5.3 p.24 |

**Nothing was rejected as out-of-scope** — no framework default outside the lecture appears, no numeric backprop item exists anywhere in the file.

## 4. PASS / FAIL per REVIEW.md criterion

| # | criterion | verdict |
|---|---|---|
| 1 | Correctness (keys re-verified with python3) | **PASS** — all 59 keys correct; no key changed (the two edits were to distractor text/rationale, not to answers) |
| 2 | Single correct answer / no ambiguity | **PASS** after the L5-37 fix (its old option 2 was not a second answer, but it was self-refuting) |
| 3 | Not beyond what was taught | **PASS** — see §3 audit; every API fact traced to a slide or notebook cell |
| 4 | Not too hard (≤3 steps, <1 min by hand, no double negatives) | **PASS** — worst case is L5-55 (3 steps: 60,000→48,000→750→×40) and L5-35 (3 products); all arithmetic is hand-doable |
| 5 | Not too easy / not pure trivia | **PASS** — the 19 `easy` items each drill a definition, a default or a coloured phrase; none is answerable by odd-one-out alone |
| 6 | Difficulty labels honest, mix ≈30/50/20 | **PASS** — 19 easy (32%) / 30 medium (51%) / 10 hard (17%). See "borderline" note below |
| 7 | Options: exactly 4, position-independent, length ratio, same form, real near-miss | **PASS** — script-checked: 0 length violations (longest/shortest ≤1.6× after HTML stripping, or all options ≤28 chars); 0 forbidden phrases ("ถูกทุกข้อ"/"ไม่มีข้อใดถูก"/references to other options) |
| 8 | Explanation teaches; `why_wrong[i]` names a misconception; `ref` real | **PASS** — all 59 have 3-element aligned `why_wrong`; all `ref`s point to a real §, slide page, or notebook cell |
| 9 | No duplicates / near-duplicates | **PASS** — Jaccard scan found only two intentional pairs (below) |
| 10 | Coverage of every section + every ex3 step | **PASS** after the 8 additions — full map below |

## 5. Coverage map (after additions)

§1 → L5-01/02/03 · §1.1 → L5-04 · §1.2 → L5-05, **L5-57**, L5-06 · §2 → L5-08 · §2.1 → L5-07 ·
§2.2 → L5-09 · §2.3 → L5-10/11/12 · §2.4 → L5-13 · §2.5 → L5-14 · §3.0 → L5-15 · **§3.0B → L5-56** ·
§3.1 → L5-16/17 · §3.2 → L5-18/19/20, **L5-58** · §3.3 → L5-21/22 · §3.4 → L5-23 · §3.5 → L5-24/25 ·
§4 → L5-26/27/28, **L5-59** · §5.0 → L5-29, **L5-54** · §5.1 → L5-30, **L5-53** · **§5.3 → L5-52** ·
§5.5 → L5-31 · §5.6 → L5-32 · §5.7 → L5-33 · §5.8 → L5-34 · §5.9 → L5-38 · §5.10 → **L5-55** ·
§5.11 → L5-39 · §5.12 → L5-42 · §6 → L5-44 · §6.1 → L5-45 · §6.2 → L5-46 · §7.1 → L5-43 · §7.2 → L5-49/50/51

ex3 steps: (1) shapes/vectorize → L5-47 · (2) build+train → L5-35, L5-36, L5-43 · (3) evaluate → L5-37 ·
(4) improvement: curves → L5-39, best epoch → L5-40, L2 → L5-49, dropout → L5-50, predict/threshold → L5-48.
MNIST notebook: reshape/scale → L5-31 · labels → L5-32 · summary → L5-34 · retrain-must-rebuild → L5-41 · predict/argmax axis → L5-42.

## 6. Borderline-but-kept

- **L5-34 vs L5-35** (Jaccard 0.33) — same counting *rule*, deliberately different architectures and numbers (13,002 vs 160,305). The lecture drills this formula on both datasets; kept as repetition-by-design.
- **L5-43 vs L5-44** (Jaccard 0.36) — two different rows of the slide-34 table (binary vs multiclass-multilabel). The lesson names swapping single-label/multilabel as "กับดักที่ข้อสอบชอบใช้มากที่สุด", so both rows deserve an item. Kept.
- **L5-50 vs L5-51** — L5-50 keys on *train-only behaviour at evaluate*, L5-51 on *rate vs keep_prob*. Their distractor sets overlap on one fact. Kept: both are separately flagged coloured-text traps.
- **L5-09 (`hard`) and L5-06 (`hard`)** — each is a 2–3 step transfer to a new case (400 inputs; a new gradient vector). Arguably `medium`, but they satisfy the spec's "apply to a new small case" wording; kept to hold the hard share at 17%.
- **L5-11** asks Xavier for 784→16 (0.0025) with He (0.00255) as the near-miss. Discriminating 0.0025 from 0.00255 is fine without a calculator (2/800 vs 2/784), but it is the tightest numeric pair in the file. Kept — it is exactly the fan_in-vs-fan_in+fan_out confusion the slides warn about.
- **L5-37 options 0 vs 1** differ only in the rounding verb (ปัดขึ้น/ปัดลง) around the same number 782. Option 1 is self-inconsistent (781.25 rounds *down* to 781), so it is eliminable by arithmetic, and the item still teaches the ceiling rule. Kept.

## 7. Unfixable / notes for the author

- None. No item had to be deleted, and no claim in the bank conflicts with the lesson or the notebooks.
- One upstream nuance worth knowing (not an error in any item): the lesson correctly flags that the slide's Glorot-uniform `limit = 6/(fan_in+fan_out)` is missing its square root. No item keys on that uniform formula, so the bank sidesteps the ambiguity — if an item on it is ever added, key it to `√(6/(fan_in+fan_out))` per the lesson's exam advice.

## 8. Final validation run

```
json.load OK, items: 59, unique ids, answer==0 everywhere,
4 options each, 3-element why_wrong each, field set unchanged (+ optional `code`),
option-length ratio violations: 0, forbidden-phrase hits: 0,
difficulty mix: easy 19 (32%) / medium 30 (51%) / hard 10 (17%),
type mix: concept 29 / code 17 / calc 13, source: lesson 44 / ex3 10 / mnist_notebook 5.
```
