# Review — q_L6.json (Lesson 6: Callbacks, TensorBoard, Hyperparameter Tuning)

Ground truth used: `0006-callbacks-and-hyperparameter-tuning.html.txt` (all 33 slide pages) +
`deep_lect7_Callbacks_with_circle_tensorflow68_inclass.ipynb`,
`deep_lect7__circle_tensorboard_inclass.ipynb`,
`deep_lecture7_movie_review_parameter_tuning_inclass.ipynb`.

**Before:** 48 items · **After:** 52 items (4 added, 0 removed) · 7 items edited.

---

## 1. Table of changes

| id | problem class | what was wrong | fix applied |
|---|---|---|---|
| L6-04 | explanation quality | `why_wrong[1]` justified option "25 บรรทัด" as "150/32 = 5 batch ของ train" but 5 epochs × 5 batches = 25 only if the `testing start` line is dropped — the stated arithmetic did not reach 25 | reworded to the exact slip: `⌈150/32⌉ = 5` batches × 5 epochs, using the train batch count instead of the validation one |
| L6-05 | `ref`/stem accuracy | stem said "สไลด์หน้า 2" but the correct option quotes *training state = weights + epoch number + optimizer state*, which is page 9 | stem now cites "หน้า 2 และ 9" (matching the existing `ref`) |
| L6-11 | distractor integrity | options 251 / 271 / **285**; `why_wrong` rationales did not produce those numbers (285 was justified as "30+110+110+11+24" — the +24 is arbitrary; "ลืม bias ของ Output 1 ตัว" gives 260 not 251) | options now **261 / 230 / 251 / 271**; 230 = classic "forgot all biases" (20+100+100+10), 251 = missed one hidden layer's 10 biases, 271 = double-counted 10 biases. All three `why_wrong` now reproduce their number exactly |
| L6-26 | `why_wrong` misalignment | `why_wrong[1]` and `why_wrong[2]` were **swapped**: [1] (which must explain option "20") talked about batch size 40, [2] (which must explain "40") talked about 800/20 | swapped back; now [1]→20 (epoch count / 800÷40), [2]→40 (800÷20) |
| L6-31 | `why_wrong` wrong misconception | `why_wrong[0]` said option `−0.600 …` came from "dividing by the range (max−min = 6)" — that gives ±0.5/±0.167. The option is actually (z−μ)/σ² = (z−5)/5 | rewritten: "หารด้วย σ²=5 แทนที่จะหารด้วย √σ²=√5" (verified in python) |
| L6-35 | distractor integrity | option **45,571** corresponded to no stated or plausible misconception (it is just 911,430⁄20); `why_wrong[2]` claimed it came from "not squaring units/dropout for the 2-layer case", which actually yields 7,380 | option replaced with **7,380** = (738+738)×5, and `why_wrong[2]` now shows that arithmetic |
| L6-27 | difficulty label honesty | labelled `easy` but requires a 3-step mechanism argument (all-positive weights → z large → tanh saturates → g′≈0) and the distractors need knowledge of tanh's range and ReLU dying units | relabelled `medium` |

### Items added (coverage gaps)

| id | fills gap | diff | source |
|---|---|---|---|
| **L6-49** | Slide **p.21** *Number of hidden layers and number of neurons* — "start `large enough`, observe, then fine-tune down"; the lesson explicitly flags "เริ่มเล็กที่สุดแล้วค่อยเพิ่ม" as the exam trap. No item covered p.21 at all | medium / concept |
| **L6-50** | Slide **p.23** *Strategies to avoid overfitting* — the tool↔knob pairing (L2→lambda, Dropout→dropout rate, BN→mini-batch size/momentum, Data augmentation→random transformations). p.23 was entirely uncovered | easy / concept |
| **L6-51** | Slide **p.29 ❶❷ + p.30** — `build_model(hp)` must take `hp` and return a **compiled** model, and the tuner takes the *function* with no parentheses; four tuner classes named in the explanation. Previously only `hp.Float`/`hp.Int` (L6-42) and `tuner.search` (L6-44) were covered | easy / code |
| **L6-52** | Slides **p.25–26** Grid vs Random *Layout* figures — the "same 9-trial budget, 3 distinct values vs 9 distinct values of the important parameter" argument. Grid/Random were only covered by a size calculation (L6-35) and a recall item (L6-36) | medium / calc |

---

## 2. Numbers re-verified with python3

Every `calc` item was recomputed. All keyed answers were already correct; no key changed.

| item | claim | recomputed |
|---|---|---|
| L6-02 | 13 printed lines | `1 + ⌈150/64⌉·3 + 3 = 1+9+3 = 13` ✓ |
| L6-04 | 15 lines | `5 · (1 + ⌈50/32⌉) = 5·3 = 15` ✓ (notebook cell 17 prints exactly this) |
| L6-07 | stop at epoch 34 (patience=3, min_delta=0.001) | simulator → `(stop 34, best 31)` ✓ |
| L6-08 | restore epoch 28 (min_delta=0.01, patience=5) | simulator → `(stop 33, best 28)` ✓ |
| L6-11 | 261 trainable | `30+110+110+11 = 261` ✓ = notebook `Total params: 261` |
| L6-12 | 524 optimizer params | `2·261 + 2 = 524`, `261+524 = 785` ✓ = notebook `load_model` summary |
| L6-15 | `my_model.2.3` / `my_model.4.3` | cumulative batches 6, 12 with 3 batch/epoch ✓ = notebook `!ls checkpoint` |
| L6-16 | 6 epochs | `10 − 4 = 6` ✓ = notebook `len(history.history['loss']) → 6` |
| L6-18 | 210 | `20·10 + 10` ✓ |
| L6-20 | 0.019683 at epoch index 3 | compounding chain `0.03 → 0.027 → 0.0243 → 0.019683 → 0.01594323` ✓ = notebook verbose log |
| L6-21 | lr stays 0.03 with decay_steps=10000 | `⌊(1+e)/10000⌋ = 0 ∀ e<9999` ✓ |
| L6-22 | 0.001 after 3 reductions | `0.03 → 0.003 → 0.001 → 0.001` ✓ |
| L6-26 | 25 steps | `⌈(1000−200)/32⌉ = 25` ✓ = notebook progress bar `25/25` |
| L6-31 | −1.342, −0.447, 0.447, 1.342 | μ=5, σ²=5 ✓ |
| L6-35 | 911,430 | `(41·3·6 + 41²·3·6²)·5 = (738+181548)·5` ✓; new distractor 7,380 = `(738+738)·5` ✓ |
| L6-37 | (9, 9) at i=2 | `⌊81/9⌋ = 9`, `1·3² = 9` ✓ |
| L6-38 | 405 epoch, ≈16× | `5·81 = 405`, `81·81/405 = 16.2` ✓ |
| L6-40 | B wins at λ=1.2 | `A = 0.874`, `B = 1.040` ✓ (and the λ=0.2 distractor value 0.84 ✓) |
| L6-41 | 41 values | `50−10+1` ✓ |
| L6-45 | 640,129 | `10000·64+64 = 640,064`, `+65 = 640,129` ✓ = notebook `final_model.summary()` |
| L6-46 | RandomSearch 30 / Hyperband 90 trials | ✓ notebook cells 23 & 30 print `Trial 30 Complete` / `Trial 90 Complete` |

Keras API defaults cross-checked against the lesson text (no invented defaults found):
`EarlyStopping` → `min_delta=0`, `restore_best_weights=False` (lesson §2 states both explicitly);
`ModelCheckpoint` → `save_weights_only=False`, `save_best_only=False`, `mode='auto'`, `save_freq='epoch'` and integer `save_freq` counts **batches** (§3, p.6–8);
`BackupAndRestore` → `backup_dir` required, `save_freq='epoch'`, `delete_checkpoint=True` (§4, p.9);
`ReduceLROnPlateau` → `factor`, `patience`, `min_delta`, `min_lr` exactly as slide p.10;
`TensorBoard` → `histogram_freq` default 0, `write_graph`, `update_freq='epoch'` (§6, p.13);
KerasTuner → `hp.Int / hp.Float / hp.Choice / hp.Boolean` and `GridSearch / RandomSearch / HyperBand / BayesianOptimization` (p.29–30). Nothing outside this set is asserted anywhere in the bank.

---

## 3. PASS / FAIL per REVIEW.md criterion

| # | criterion | verdict | note |
|---|---|---|---|
| 1 | **Correctness** | **PASS** | All 21 calc answers recomputed in python; no key was wrong. 7 items had defective *supporting* material (see table), all fixed |
| 2 | **Single correct answer** | **PASS** | No item has a second defensible option. L6-42's `hp.Choice` distractor is the closest call (it does technically sample only 3 decade end-points) but the stem says "ระหว่าง 0.0001 ถึง 0.01", i.e. the whole range — kept |
| 3 | **Not beyond what was taught** | **PASS** | Every default/API detail traced to a slide page or a notebook cell. No numeric backprop. No un-taught framework behaviour |
| 4 | **Not too hard** | **PASS** | Max chain is 3 steps (L6-08, L6-35, L6-38). No double negatives — two items use a single "ข้อใด**ไม่**ตรงกับ" stem (L6-17, L6-33), which the spec permits |
| 5 | **Not too easy / not trivia** | **PASS** | Every `easy` item drills a stated default, a coloured-text definition or a pairing that the lesson flags as an exam trap |
| 6 | **Difficulty mix** | **PASS** | 16 easy / 26 medium / 10 hard = **31 / 50 / 19** (target 30/50/20). L6-27 relabelled to reach this honestly |
| 7 | **Options** | **PASS** | 4 options each; no "ถูกทุกข้อ"/"ไม่มีข้อใดถูก"/positional references; length ratio ≤1.6 after stripping HTML for every prose item; the 14 short numeric/term items are same-form and ≤28 chars |
| 8 | **Explanation teaches** | **PASS** | Every explanation shows the computation or quotes the coloured slide phrase via `<mark class="src" data-p="…">`; all 52×3 `why_wrong` entries now name a misconception that actually produces its option (4 were repaired) |
| 9 | **No duplicates** | **PASS** | No duplicate ids, no duplicate option text, no near-duplicate stems. Borderline pairs kept — see §4 |
| 10 | **Coverage** | **PASS after additions** | Every lesson section §1–§11 and every slide page 1–32 that carries teachable content now has ≥1 item. All three notebooks represented (callbacks 9, tensorboard 4, tuning 3) |

Invariants preserved: `answer` = 0 for all 52; `why_wrong` is a 3-element array everywhere; ids unique (`L6-01`…`L6-52`); field set unchanged (`code` optional); file is a valid JSON array (`json.load` verified).

---

## 4. Borderline-but-kept

- **L6-17 / L6-33** — negative stems ("ข้อใด**ไม่**ตรงกับ…"). Single negation only, no double negative, and both drill lists the lecturer explicitly colour-coded (BackupAndRestore's 4 rules; p.22's "epochs ไม่ต้องจูน"). Kept.
- **L6-37 / L6-38** — both work the same HyperBand bracket `s=4`. Different questions (per-round `(n_i, r_i)` vs total budget & 16× ratio) and the lesson treats both as separate exam-ready numbers. Kept.
- **L6-06 / L6-09** — both touch `restore_best_weights`. L6-06 keys on the *defaults pair* (`min_delta=0`, `restore=False`); L6-09 keys on *reading the p.3 evaluate table* (0.5149 vs 0.4750). Kept.
- **L6-42 option [3]** (`hp.Choice("lr", values=[1e-4, 1e-3, 1e-2])`) — arguably gives "equal chance per order of magnitude". The stem asks for selection *between* 0.0001 and 0.01, which Choice cannot do; `why_wrong[2]` says so explicitly. Kept as the strong near-miss.
- **L6-22** uses the slide's `min_delta` context while the in-class notebook ran `min_delta=0.01`; the stem does not mention `min_delta` at all, so both sources agree on the keyed answer. Kept.
- **L6-45** asks about the notebook's 64-unit final model (640,129) while §8 of the lesson works the 32-unit refined winner (320,065). Both numbers are real and in the sources; the stem names the architecture explicitly, so there is no ambiguity. Kept.

## 5. Nothing left unfixed

No item was found that could not be repaired in place. No item was removed.
