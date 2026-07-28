#!/usr/bin/env python3
"""LLM agent driver for the Curling Arena (Groq OpenAI-compatible API).

Plays full matches against the resident champion through the arena's public
agent protocol: reads the plain-text state, lets the LLM think + optionally
preview one shot via /solve, then commits a throw. The API key comes from the
GROQ_API_KEY env var (never written to disk).

  GROQ_API_KEY=... python3 arena/agents/groq_agent.py \
      --model openai/gpt-oss-120b --ends 8 --matches 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8020")
ap.add_argument("--model", default="openai/gpt-oss-120b")
ap.add_argument("--ends", type=int, default=8)
ap.add_argument("--matches", type=int, default=1)
ap.add_argument("--side", default="A")
ap.add_argument("--temperature", type=float, default=0.5)
ap.add_argument("--max-previews", type=int, default=1, help="solve previews the LLM may request per turn")
ap.add_argument("--log-dir", default=str(Path(__file__).parent / "logs"))
args = ap.parse_args()

KEY = os.environ.get("GROQ_API_KEY")
assert KEY, "set GROQ_API_KEY"
SIDE, OPP = args.side, ("B" if args.side == "A" else "A")
LOGDIR = Path(args.log_dir)
LOGDIR.mkdir(parents=True, exist_ok=True)

SYSTEM = f"""You are an expert curling skip playing MIXED DOUBLES curling as Team {SIDE} through a JSON API.
You will receive the full match state as text each turn (score, hammer, every stone's coordinates, whose turn).

COORDINATES: [along, lateral] in meters. along=0 is the button (center of the scoring house, radius 1.83);
NEGATIVE along = in front of the house (guard zone); positive = behind the button. lateral: + is right.
Stones are REMOVED if they finish past the back line (along > +1.97) or touch a side board (|lateral| > 2.23).
Execution noise is ON: your throw lands near, not exactly on, your intent. Robust shots beat perfect ones.

STRATEGY BASICS: only stones touching the house score; closest stone to the button wins the end for its team,
one point per stone closer than the opponent's best. With hammer (throwing last), keep the center open and
score 2+; without hammer, put guards up and try to steal. The no-takeout rule forfeits throws that remove an
OPPONENT stone while 'RULE ACTIVE' is shown — move them without removing, or play your own stones.
Takeouts are legal and strong afterwards. Freezes (draw touching an enemy stone on the button side) are
hard to remove. Corner guards + draws behind them win ends.

RESPOND WITH ONE JSON OBJECT ONLY, no prose outside it:
{{"thought": "<=25 words",
  "command": "preview" | "throw" | "powerplay" | "pass",
  "shot": {{"type": "draw", "target": [along, lateral]}}
        | {{"type": "contact", "target": [along, lateral], "weight": "soft|medium|heavy"}}
        | {{"type": "after_contact", "stone_slot": K, "target": [along, lateral]}}
        | {{"type": "after_contact", "stone_slot": K, "remove": true}},
  "wing": "left" | "right"}}
- "preview": test the shot first; you get the predicted result and must then decide again (max {args.max_previews}/turn).
- "throw": commit the shot.
- "powerplay"/"pass": ONLY when the state says you may call your power play (include "wing" for powerplay).
Slot numbers K come from the state text ("slot N"). Prefer draw / contact / after_contact intents;
they are solved into physics for you and the solver reports how achievable they are."""


def api(path, method="GET", body=None, timeout=900):
    req = urllib.request.Request(args.base + path,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"}, method=method)
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout)), None
    except urllib.error.HTTPError as e:
        try:
            return None, json.loads(e.read().decode()).get("detail", str(e))
        except Exception:
            return None, str(e)


_last_llm_call = [0.0]
MIN_INTERVAL = 4.0   # pace requests to stay under free-tier RPM limits


def llm(messages):
    body = {"model": args.model, "messages": messages, "temperature": args.temperature,
            "max_tokens": 4096, "response_format": {"type": "json_object"}}
    if "gpt-oss" in args.model:
        body["reasoning_effort"] = "low"   # curling turns don't need long chains; saves TPM
    data = json.dumps(body).encode()
    for attempt in range(8):
        wait = MIN_INTERVAL - (time.time() - _last_llm_call[0])
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",
                                     data=data,
                                     headers={"content-type": "application/json",
                                              "authorization": f"Bearer {KEY}",
                                              "user-agent": "curl/8.5.0"})  # CF blocks urllib UA
        try:
            out = json.load(urllib.request.urlopen(req, timeout=120))
            _last_llm_call[0] = time.time()
            txt = out["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0) if m else txt)
        except urllib.error.HTTPError as e:
            _last_llm_call[0] = time.time()
            if e.code == 429 and attempt < 7:
                retry = e.headers.get("retry-after")
                body_txt = ""
                try:
                    body_txt = e.read().decode()[:200]
                except Exception:
                    pass
                m = re.search(r"try again in ([0-9.]+)s", body_txt)
                delay = float(retry) if retry else (float(m.group(1)) if m else 10.0 * (attempt + 1))
                time.sleep(min(delay + 1.0, 120.0))
                continue
            if attempt == 7:
                raise
            time.sleep(5 * (attempt + 1))
        except Exception:
            if attempt == 7:
                raise
            time.sleep(5 * (attempt + 1))


def shot_body(shot):
    b = {"side": SIDE, "type": shot.get("type", "draw")}
    for k in ("target", "weight", "stone_slot", "remove", "action"):
        if shot.get(k) is not None:
            b[k] = shot[k]
    return b


def play_match(mi):
    out, err = api("/api/match", "POST", {
        "players": {SIDE: "agent", OPP: "champion"},
        "labels": {SIDE: args.model, OPP: "champion az_v14d"},
        "ends": args.ends, "first_hammer": "random"})
    assert not err, err
    mid = out["match"]["id"]
    log = (LOGDIR / f"{args.model.replace('/', '_')}_{mid}.jsonl").open("a")

    def note(kind, **kw):
        log.write(json.dumps({"t": time.strftime("%H:%M:%S"), "kind": kind, **kw}) + "\n")
        log.flush()

    note("match_start", mid=mid, model=args.model, ends=args.ends)
    print(f"[match {mi}] id={mid} model={args.model}", flush=True)
    llm_calls = 0
    history = []   # rolling feedback for the LLM

    while True:
        state, err = api(f"/api/match/{mid}")
        assert not err, err
        m = state["match"]
        if m["status"] != "in_progress":
            note("match_over", totals=m["totals"], winner=m["winner"],
                 ends=[e["score"] for e in m["ends"] if e.get("score")], llm_calls=llm_calls)
            print(f"[match {mi}] OVER: A {m['totals']['A']} : {m['totals']['B']} B  winner={m['winner']}",
                  flush=True)
            return m
        turn = m["turn"]
        text = state["text"]

        if turn["team"] != SIDE:
            # champion on turn: power-play window (or opening). Ask about powerplay if offered.
            if "power play" in text and f"Team {SIDE}" in text and "WAITING" in text:
                d = llm([{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text + "\nDecide: powerplay or pass."}])
                llm_calls += 1
                if d.get("command") == "powerplay" and d.get("wing") in ("left", "right"):
                    _, e2 = api(f"/api/match/{mid}/powerplay", "POST",
                                {"side": SIDE, "wing": d["wing"]})
                    note("powerplay", wing=d.get("wing"), err=e2, thought=d.get("thought"))
            _, err = api(f"/api/match/{mid}/champion_move", "POST", {})
            if err:
                note("champion_move_err", err=err)
                time.sleep(2)
            continue

        # --- our turn ---
        messages = [{"role": "system", "content": SYSTEM}]
        fb = ("\n\nRecent feedback:\n" + "\n".join(history[-3:])) if history else ""
        messages.append({"role": "user", "content": text + fb +
                         "\n\nYour move. Respond with the JSON object only."})
        previews = 0
        while True:
            d = llm(messages)
            llm_calls += 1
            cmd = d.get("command", "throw")
            shot = d.get("shot") or {}
            if cmd == "preview" and previews < args.max_previews and shot:
                res, err = api(f"/api/match/{mid}/solve", "POST", shot_body(shot))
                previews += 1
                if err:
                    messages.append({"role": "assistant", "content": json.dumps(d)})
                    messages.append({"role": "user", "content": f"Preview rejected: {err}. Decide again (throw)."})
                    continue
                info = {"solver": res["solver"],
                        "illegal": res["preview"]["illegal_takeout"],
                        "champion_eval_after (A persp)": res["preview"]["predicted_value_A"]}
                note("preview", shot=shot, info=info, thought=d.get("thought"))
                messages.append({"role": "assistant", "content": json.dumps(d)})
                messages.append({"role": "user", "content":
                                 f"Preview result: {json.dumps(info)}. Now commit: respond with a throw JSON "
                                 f"(same shot or adjusted)."})
                continue
            # commit
            res, err = api(f"/api/match/{mid}/throw", "POST", shot_body(shot))
            if err:
                note("throw_rejected", shot=shot, err=err)
                messages.append({"role": "assistant", "content": json.dumps(d)})
                messages.append({"role": "user", "content": f"Throw rejected: {err}. Fix and respond again."})
                continue
            rec = res["result"]["throw"]
            solver = res.get("solver", {})
            note("throw", n=rec["n"], end=rec["end"], shot=shot, solver=solver,
                 illegal=rec["illegal_takeout"], value_A=rec.get("value_A"),
                 thought=d.get("thought"),
                 replies=[r["throw"]["n"] for r in res.get("replies") or []],
                 end_result=res["result"].get("end_result") or
                            next((r.get("end_result") for r in (res.get("replies") or [])
                                  if r.get("end_result")), None))
            err_m = solver.get("achieved_error_m")
            history.append(
                f"end {rec['end']} throw {rec['n']}: you played {json.dumps(shot)} -> "
                f"{'ILLEGAL takeout, forfeited' if rec['illegal_takeout'] else 'ok'}"
                + (f", solver landed {err_m} m off your target" if err_m is not None else "")
                + (f", champion eval after your shot (A persp): {rec.get('value_A')}" if rec.get("value_A") is not None else ""))
            print(f"[match {mi}] end {rec['end']} throw {rec['n']}: {shot.get('type')} "
                  f"vA={rec.get('value_A')}", flush=True)
            break


results = []
for i in range(args.matches):
    results.append(play_match(i + 1))
print("\n=== RESULTS ===")
for m in results:
    ends = [f"E{k+1}:{(e['score']['team'] or '-') }+{e['score']['points']}"
            for k, e in enumerate(m["ends"]) if e.get("score")]
    print(f"{args.model} (Team {SIDE}) vs champion: A {m['totals']['A']} : {m['totals']['B']} B "
          f"winner={m['winner']}  [{' '.join(ends)}]  undos={len(m.get('undos', []))}")
