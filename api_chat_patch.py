"""
Drop-in replacement for the /api/chat section in app.py.

WHAT CHANGED AND WHY
---------------------
1. All counting/aggregation now happens in Python (build_chat_context),
   not inside the LLM's head. The model is handed final numbers, not
   raw per-hour/per-day arrays it has to sum itself. This is why
   "how many helmet violations" was failing — it was trying to add up
   a hourly_breakdown / by_type list buried in a giant f-string.

2. Data is sent as clean json.dumps(...) instead of Python dict repr
   (f"{stats}"). Dict reprs use single quotes and Python-specific
   syntax that models parse less reliably than JSON.

3. The system prompt is cut from ~250 lines to ~40. Every rule that
   was "just in case" (formal report avoidance, exhaustive good/bad
   example blocks, a whole meta-section teaching the model what an
   internal checklist looks like) was removed. Long, example-heavy
   prompts on small/fast models increase the odds of the model
   echoing the *shape* of an example (e.g. "Check Constraints:")
   instead of following the instruction around it. Short, direct
   prompts are far more reliable for exactly this class of bug.

4. vtype keys are normalized (lowercase, spaces->underscores) so
   "helmet violations" reliably matches "no_helmet" / "No Helmet" /
   etc. regardless of how it's stored in the DB.
"""

import json
from flask import request, jsonify
from database import get_comprehensive_analytics, get_stats


def _norm_type_counts(records):
    """[{'type': 'no_helmet', 'count': 3}, ...] -> {'no_helmet': 3, ...}"""
    out = {}
    for r in records:
        key = str(r["type"]).strip().lower().replace(" ", "_")
        out[key] = out.get(key, 0) + r["count"]
    return out


def build_chat_context(cameras: dict):
    """Pre-aggregate everything the chatbot needs into one small,
    already-computed JSON-able dict. No raw hourly/daily arrays —
    only totals, breakdowns, and peaks the model can quote directly."""
    analytics = get_comprehensive_analytics()
    stats = get_stats()

    camera_info = [
        {
            "name": c.name,
            "location": c.location,
            "online": c.online,
            "vehicles_now": c.vehicle_count,
        }
        for c in cameras.values()
    ]

    daily = analytics["daily"]
    weekly = analytics["weekly"]
    monthly = analytics["monthly"]

    return {
        "today": {
            "date": daily["date"],
            "total_confirmed": daily["total_violations"],
            "by_type": _norm_type_counts(daily["by_type"]),
            "by_camera": {
                c["cam_id"]: c["count"] for c in daily["by_camera"]
            },
            "peak_hour": daily["peak_hour"],
        },
        "this_week": {
            "range": f"{weekly['week_start']} to {weekly['week_end']}",
            "total_confirmed": weekly["total_violations"],
            "by_type": _norm_type_counts(weekly["by_type"]),
            "peak_day": weekly["peak_day"],
        },
        "this_month": {
            "month": monthly["month"],
            "total_confirmed": monthly["total_violations"],
            "by_type": _norm_type_counts(monthly["by_type"]),
        },
        "pending_reviews_all_time": stats["pending_reviews"],
        "cameras": camera_info,
    }


SYSTEM_PROMPT_TEMPLATE = """You are TrafficGuard AI, the chat assistant inside the TrafficGuard \
traffic violation dashboard.

DATA (only source of truth — never invent numbers):
{data_json}

Violation types tracked: no_helmet, wrong_direction, triple_riding, red_light. \
(Overspeeding is not tracked — don't mention it.)

RULES
1. Answer the user's question using only the DATA above. If a number isn't \
in DATA, say you don't have it — don't estimate.
2. "by_type" keys already give you the exact count per violation type — \
just look it up, don't recompute from anything else.
3. pending_reviews_all_time are NOT confirmed violations — never call them \
"violations" outright.
4. After answering, add one short insight only if the data supports it \
(dominant type, high-traffic camera, notable peak). Skip it if there's \
nothing meaningful to say.
5. End with exactly one relevant follow-up question. Never zero, never more \
than one, never a generic "anything else?".
6. Keep it short: 1-3 sentences for simple questions, up to ~6 short lines \
for anything broader. Use **bold** for key numbers. No headings, no tables, \
no long reports unless explicitly asked.
7. Output ONLY the final message to the user. Never show your reasoning, \
checklists, labels like "Direct answer:"/"Insight:"/"Check constraints:", or \
any other meta-commentary — just the natural reply itself.
8. If asked something unrelated to TrafficGuard/traffic safety, say: \
"I'm focused on TrafficGuard and traffic analytics, so I can't help much \
with that." then optionally one relevant redirect question.

Example (style only, not real data):
User: "How many helmet and wrong-side violations today?"
Reply: "Today there are **4 helmet violations** and **2 wrong-direction \
violations**. Helmet violations are the bigger share right now. Want me to \
check which camera is catching most of them?"
"""


def api_chat(client, cameras):
    """Call this from your Flask route:

        @app.route("/api/chat", methods=["POST"])
        def _api_chat():
            return api_chat(client, cameras)
    """
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"reply": "Please enter a message."}), 400

    context = build_chat_context(cameras)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        data_json=json.dumps(context, indent=2)
    )

    try:
        from google.genai import types

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=800,
                temperature=0.4,
            ),
        )
        return jsonify({"reply": response.text})

    except Exception as e:
        print("========== GEMINI ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("==================================")
        return jsonify({
            "reply": "Gemini API error. Check the Flask terminal.",
            "error": str(e),
        }), 500