# Review — q_L1.json (Lesson 1: ML Foundations + homework ex1)

Reviewed: 49 items in, **53 items out** (1 item repaired, 4 items added, 0 removed).
Ground truth: `0001-machine-learning-foundations.html.txt` (read in full, §0–§12 + page-coverage appendix)
and `deep-learning/ex1_6880318026.ipynb` (all 31 cells, code + stored outputs).
Every numeric key was re-derived with `python3` (see "Numeric re-verification" below).

## Items changed

| id | problem class | what I did |
|---|---|---|
| L1-37 | **Two options meaning the same thing** (REVIEW #2). Options[1] `g = 1/(1+np.exp(z))` and options[3] `g = np.exp(-z)/(1+np.exp(-z))` are algebraically identical — both equal \(g(-z)=1-g(z)\) (verified numerically). With two equally-wrong-for-the-same-reason options the item no longer discriminates, and a student who spots the identity can eliminate both for free. | Replaced options[3] with `g = 1/(1+math.exp(-z))` — the formula is right but `math.exp` raises `TypeError: only 0-dimensional arrays can be converted to Python scalars` on the `np.arange(-8,8,0.01)` array the next cell feeds it. This is the *course's own* stated ex1 objective ("ประยุกต์ใช้ Vectorization ในการลดการ For Loop"), so it is a real misconception rather than nonsense. Rewrote `why_wrong[2]` accordingly and expanded `explanation` to show the hand-check \(g(0)=0.5,\ g(2)=0.881\) and why `np.exp` is required. Option lengths improved from 20/19/20/29 (ratio 1.53) to 20/19/20/22 (ratio 1.16). |

## Items added (coverage gaps, REVIEW #10)

| id | gap it fills | difficulty / type |
|---|---|---|
| L1-50 | **§1 p.12–13 had no item.** Items L1-04/L1-05 covered p.14 (why DL is popular *today*) and p.19 (challenges), but the two coloured headline reasons *why DL is powerful* — "Scale for complex tasks" and "Automate feature engineering (i.e., representation learning)" — were untested, even though the lesson flags both as high-yield with named traps ("simple network", "learning features manually"). Explanation walks the real numbers (100×100 RGB = 30,000 features) and the 3-box ML vs 2-box DL Car/Not-Car diagram. | medium / concept |
| L1-51 | **§5 p.39 had no item.** The `1 = positive class (finding something)` / `0 = negative class (absenting something)` definition and the "positive ≠ good, spam = 1" trap were untested, although the lesson calls this out as a prerequisite for reading precision/recall correctly in §7. | easy / concept |
| L1-52 | **§7 accuracy-on-imbalanced-data trap untested.** L1-22/23/24 cover precision/recall arithmetic, the threshold trade-off and ROC, but the lesson's explicit warning ("accuracy หลอกได้เมื่อข้อมูลไม่สมดุล — 1 positive in 100 ⇒ accuracy 99% but recall 0") had no item. Verified: TP=0, FP=0, FN=1, TN=99 ⇒ accuracy 0.99, recall 0.0. Explanation notes precision is *undefined* here (TP+FP=0), so the stem deliberately asks only for accuracy and recall. | hard / calc |
| L1-53 | **§2 p.28's worked example untested.** The lesson's own "ชัยชนะของบทเรียนนี้" checklist names "การแก้ θ จากตารางบ้าน 3 แถวจนได้ \(h=10{,}000x_1\)" as a must-be-able-to-do, and it is the source of the "θ_j = 0 means the feature doesn't help" idea that §11 builds on. Verified with `np.linalg.solve` ⇒ θ = [0, 10000, 0]. The row-3-minus-row-1 trick keeps it to 3 steps / well under a minute. | hard / calc |

## Borderline, kept as-is

| id | why it looked suspect | why I kept it |
|---|---|---|
| L1-03 | Pure date recall (1989 / 1943 / 2010 / 2014) — flirts with REVIEW #5 "pure trivia". | The lesson explicitly lists this as a coloured-text exam trap ("กับดักคือสลับ 2014 กับ 2017 หรือย้าย backpropagation ไปปี 1943"), and all four options are the same form (4-digit years), so it is not answerable by odd-one-out. Correctly labelled `easy`. |
| L1-18, L1-19 | Explanations and options contain raw `<` inside KaTeX (`z=-1<0`, `h<0.5`). | HTML never treats `<` followed by a digit as a tag start, so it renders as text and KaTeX parses it. The same pattern already exists in the sibling `q_L4.json`, so changing it here alone would break consistency. Flagged, not changed. |
| L1-26, L1-32 | Labelled `hard` but each is arguably 2–3 quick steps. | Both match the spec's definition of `hard` ("apply to a new small case, max 3 steps"): L1-26 needs softmax-of-equal-logits → 0.25 → \(-\ln 0.25 = \ln 4 = 2\ln 2 = 1.386\); L1-32 needs skip-θ₀ → square → ×(λ/m). Verified 2.5 and 1.386. |
| L1-31 | Answer is a verbatim slide sentence ("More examples does help in high variance case!"), so it reads `easy` rather than `medium`. | The discrimination is in the *direction* of the bias/variance pairing, which the lesson calls "ข้อที่พลาดกันบ่อยที่สุดในหัวข้อนี้" — that is comparison, not recall. Kept `medium`; keeping it also holds the overall mix at 32/49/19. |
| L1-34 | Longest/shortest option ratio 1.47 — the highest in the file. | Under the 1.6 limit, all four options are the same grammatical form (a named quantity + its role), and the short option ("degree ของ polynomial") is a *distractor*, so length gives no hint toward the key. |
| L1-42, L1-44 | Each has one distractor that is close to nonsense (epsilon "acts as learning rate"; "cost = class ratio × 1.4"). | SPEC explicitly allows one clearly-eliminable option per item ("it's fine to have one that is clearly eliminable if you truly understand"); the other two distractors in both items are genuine near-misses. |
| L1-48 | `scipy.optimize.minimize(..., jac=True, method='TNC')` could look like framework trivia the course never taught (REVIEW #3). | It is verbatim ex1 cell (28), which is listed ground truth, and the item asks what `jac=True` *means* rather than an undocumented default. |
| §0 (p.2–4, grading policy) | REVIEW #10 asks every major section to have ≥1 item, and the lesson flags "Midterm 30% / Final 40% / Homework 20% / Assignment 10%" as a trap. | Deliberately **not** added: this is course-administration memorisation with zero ML content, which REVIEW #5 rules out as pure trivia. Recorded here so the omission is a decision, not an oversight. |
| §3.1 MSE vs MAE | The slide's MAE box has no dedicated item. | L1-10 tests MSE arithmetic and uses MAE (=1.5) as its near-miss distractor, with the distinction explained in `explanation` and `why_wrong[0]`. Judged covered. |

## Numeric re-verification (all with `python3`)

`L1-09` 36 (distractors 26/33/31) · `L1-10` MSE 2.5, MAE 1.5 · `L1-12` 3−0.1(6)=2.4 (distractors 3.6/2.7/2.1)
· `L1-15` 20/80=0.25 (0.40/0.20/0.50) · `L1-19` z(2,2)=−1 · `L1-20` −ln0.1=2.303 · `L1-22` 0.75 / 0.60, acc 0.70, F₁ 0.667
· `L1-26` a=0.25, −ln0.25=1.386, 4×=5.545 · `L1-29` 500/5=100, 400 · `L1-32` 0.5×5=2.5 (j=0 ⇒ 15.0; no-square ⇒ 1.5; no λ/m ⇒ 5.0)
· `L1-35` (3+8+10)/3=7.0 (5.5/9.0) · `L1-38` σ(0)(1−σ(0))=0.25 · `L1-52` acc 0.99 / recall 0.0 · `L1-53` θ=[0, 10000, 0].
Notebook re-run under `np.random.seed(10)`: `randInitializeWeights(2,1)` → `[0.0651, −0.1150, 0.0321]` and
`nnCostFunction` → `(0.7141475725196582, [0.00590843, −0.25222832, −0.25222832])` — exactly the stored outputs quoted by
L1-39 and L1-44, and `z1` shape (600,3)@(3,1)=(600,1) for L1-40.

## PASS / FAIL per REVIEW criterion

| # | Criterion | Verdict |
|---|---|---|
| 1 | Correctness (every number re-verified in python3) | **PASS** — 53/53. No wrong keys found. |
| 2 | Single correct answer, no two options meaning the same thing | **PASS after fix** — 1 failure (L1-37) found and repaired; automated pairwise option-identity check now clean. |
| 3 | Not beyond what was taught | **PASS** — every item traces to a slide page in the lesson text or to an ex1 cell; no numeric backprop, no untaught framework defaults. |
| 4 | Not too hard (≤3 steps, ~20s stem, no double negatives) | **PASS** — longest chains are L1-52/L1-53 at 3 steps each; no double negatives or trick-on-trick found. |
| 5 | Not too easy / not pure trivia | **PASS** — the two closest calls (L1-03 years, §0 grading %) are documented above; the grading-% item was deliberately not written. |
| 6 | Difficulty label honest; mix ≈30/50/20 | **PASS** — final mix **17 easy / 26 medium / 10 hard = 32.1 / 49.1 / 18.9 %**. One borderline label (L1-31) documented. |
| 7 | Options: exactly 4, position-independent, comparable length, same form, real near-miss distractors | **PASS after fix** — 4 options everywhere; no "ถูกทุกข้อ / ไม่มีข้อใดถูก / ข้างต้น" or cross-option references; max length ratio now **1.47** (L1-34), all others ≤1.45, none over the 1.6 limit. |
| 8 | Explanation teaches; `why_wrong[i]` names the misconception; `ref` real | **PASS** — all 53 explanations show the steps; all `why_wrong` arrays are 3 elements aligned to options[1..3]; every `ref` points to a real lesson section/page or ex1 cell number. |
| 9 | No duplicate / near-duplicate items | **PASS** — closest pairs are L1-32 (calc) vs L1-33 (concept) on L2 regularization and L1-30 vs L1-31 on bias/variance; each pair tests a different skill (compute vs classify, diagnose vs prescribe). |
| 10 | Coverage of every lesson section and ex1 step | **PASS after additions** — §0A, §1, §1A, §1B, §2, §3.1–3.3, §3A, §4, §5, §6, §7, §8, §9, §10/10A, §11, §11A, §12 all have ≥1 item; ex1 covered end-to-end (sigmoid, sigmoidGradient, randInitializeWeights, forward shapes, gradient line, epsilon, GD update, initial cost ≈ ln2, predict, data split, 70 % result, scipy `jac=True`, plotDecisionBoundary). §0 (grading policy) intentionally uncovered. |

## Invariants re-checked after editing
`json.load` succeeds · 53 items · all `answer == 0` · all `len(options) == 4` · all `len(why_wrong) == 3` ·
all ids unique · field set unchanged (`code` optional, present on 5 items) · `lesson == 1` everywhere ·
`source` ∈ {lesson (40), ex1 (13)} · `type` ∈ {concept 27, calc 16, code 10}.
