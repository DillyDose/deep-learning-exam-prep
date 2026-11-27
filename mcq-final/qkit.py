"""Shared helper for final MCQ sources: correct option first, build.py shuffles positions."""
import json, os

def make(lesson):
    def Q(i, topic, typ, diff, stem, opts, expl, why, ref, code=None, src="lesson"):
        d = {"id": f"L{lesson}-{i:02d}", "lesson": lesson, "topic": topic, "source": src, "type": typ, "difficulty": diff,
             "stem": stem, "options": opts, "answer": 0, "explanation": expl, "why_wrong": why, "ref": ref}
        if code: d["code"] = code
        return d
    return Q

def dump(lesson, qs):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"q_L{lesson}.json")
    json.dump(qs, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"L{lesson}", len(qs))
