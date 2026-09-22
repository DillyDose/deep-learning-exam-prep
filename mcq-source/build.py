#!/usr/bin/env python3
"""Validate the per-lesson question JSON banks, balance A/B/C/D answer positions, emit the review HTML."""
import json, re, sys, random, glob, os, collections

SP = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SP, "dl-mcq-review.html")
random.seed(20260923)

TYPES = {"concept", "code", "calc"}
DIFFS = {"easy", "medium", "hard"}
SOURCES = {"lesson", "ex1", "ex2", "ex3", "ex_shallow_nn", "mnist_notebook",
           "callbacks_notebook", "tensorboard_notebook", "tuning_notebook"}
BANNED = ["ถูกทุกข้อ", "ถูกทั้งหมด", "ไม่มีข้อใดถูก", "ผิดทุกข้อ", "ทั้ง ก", "ทั้งข้อ",
          "all of the above", "none of the above", "ข้อ ก.", "ข้อ ข.", "ตัวเลือก A", "ตัวเลือก B"]

def raw(s):
    """Strip HTML tags only — used for duplicate detection (keeps math/code punctuation)."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()

def plain(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\\[a-zA-Z]+", "x", s)
    s = re.sub(r"[\\(){}$]", "", s)
    return s.strip()

errors, warns = [], []
qs = []
for f in sorted(glob.glob(os.path.join(SP, "q_L*.json"))):
    try:
        data = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        errors.append(f"{os.path.basename(f)}: JSON parse failed: {e}")
        continue
    if not isinstance(data, list):
        errors.append(f"{os.path.basename(f)}: not a JSON array")
        continue
    qs.extend(data)

seen_ids, seen_stems = set(), {}
for q in qs:
    qid = q.get("id", "?")
    for k in ("id", "lesson", "topic", "source", "type", "difficulty", "stem", "options", "answer", "explanation", "why_wrong", "ref"):
        if k not in q:
            errors.append(f"{qid}: missing field '{k}'")
    if qid in seen_ids:
        errors.append(f"{qid}: duplicate id")
    seen_ids.add(qid)
    if q.get("type") not in TYPES:
        errors.append(f"{qid}: bad type {q.get('type')!r}")
    if q.get("difficulty") not in DIFFS:
        errors.append(f"{qid}: bad difficulty {q.get('difficulty')!r}")
    if q.get("source") not in SOURCES:
        errors.append(f"{qid}: bad source {q.get('source')!r}")
    if q.get("lesson") not in (1, 2, 3, 4, 5, 6):
        errors.append(f"{qid}: bad lesson {q.get('lesson')!r}")
    opts = q.get("options") or []
    if len(opts) != 4:
        errors.append(f"{qid}: {len(opts)} options (need 4)")
        continue
    if q.get("answer") != 0:
        errors.append(f"{qid}: answer must be 0 in source bank (got {q.get('answer')})")
    if len(q.get("why_wrong") or []) != 3:
        errors.append(f"{qid}: why_wrong must have 3 entries")
    if len({raw(o) for o in opts}) != 4:
        errors.append(f"{qid}: duplicate options")
    L = [len(plain(o)) for o in opts]
    ratio = max(L) / max(1, min(L))
    if ratio > 1.6 and max(L) > 28:
        errors.append(f"{qid}: option length ratio {ratio:.2f} (lens {L})")
    for o in opts:
        for b in BANNED:
            if b.lower() in plain(o).lower():
                errors.append(f"{qid}: banned option phrasing {b!r}")
    key = plain(q.get("stem", ""))[:90]
    if key in seen_stems:
        warns.append(f"{qid}: stem looks like a near-duplicate of {seen_stems[key]}")
    else:
        seen_stems[key] = qid

per_lesson = collections.Counter(q["lesson"] for q in qs)
for n in range(1, 7):
    if per_lesson[n] < 20:
        errors.append(f"Lesson {n}: only {per_lesson[n]} questions (min 20)")

if errors:
    print("VALIDATION ERRORS (%d):" % len(errors))
    for e in errors[:80]:
        print("  -", e)
    sys.exit(1)

# ---- balance correct-answer positions ----
global_pool = []
for n in range(1, 7):
    ls = [q for q in qs if q["lesson"] == n]
    base, rem = divmod(len(ls), 4)
    targets = [i for i in range(4) for _ in range(base)]
    extra = sorted(range(4), key=lambda i: (len([1 for t in global_pool if t == i]), random.random()))[:rem]
    targets += extra
    global_pool += targets
    random.shuffle(targets)
    for q, t in zip(ls, targets):
        correct = q["options"][0]
        others = q["options"][1:]
        why = list(q["why_wrong"])
        pairs = list(zip(others, why))
        random.shuffle(pairs)
        new_opts, new_why = [None] * 4, [""] * 4
        new_opts[t] = correct
        it = iter(pairs)
        for i in range(4):
            if i == t:
                continue
            o, w = next(it)
            new_opts[i], new_why[i] = o, w
        q["options"], q["answer"], q["why_wrong_map"] = new_opts, t, new_why
        q.pop("why_wrong", None)

# ---- report ----
print("Total questions:", len(qs))
print("Per lesson:", dict(sorted(per_lesson.items())))
print("Per type:", dict(collections.Counter(q["type"] for q in qs)))
print("Per difficulty:", dict(collections.Counter(q["difficulty"] for q in qs)))
print("Per source:", dict(collections.Counter(q["source"] for q in qs)))
pos = collections.Counter(q["answer"] for q in qs)
print("Answer positions A/B/C/D:", [pos[i] for i in range(4)])
for n in range(1, 7):
    p = collections.Counter(q["answer"] for q in qs if q["lesson"] == n)
    print(f"  L{n}: {[p[i] for i in range(4)]}")
if warns:
    print("WARNINGS:")
    for w in warns:
        print("  -", w)

tpl = open(os.path.join(SP, "template.html"), encoding="utf-8").read()
payload = json.dumps(qs, ensure_ascii=False).replace("</script>", "<\\/script>")
open(OUT, "w", encoding="utf-8").write(tpl.replace("__DATA__", payload))
print("Wrote", OUT, os.path.getsize(OUT), "bytes")
