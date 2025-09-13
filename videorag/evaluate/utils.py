"""Utility helpers for evaluation module.

This file was extracted from evaluate.py to hold small, focused helper
functions so they can be reused and tested independently.
"""
from typing import Any, Dict
import json
from difflib import SequenceMatcher


def format_test_input_for_prompt(eval_input: Dict[str, Any]) -> str:
    """Format the eval_input into a concise, human-readable block for the prompt.

    Produces:
    question:
    ground_truth_keypoints:
      video: [...]
      text: [...]
    model_answer:
    """
    try:
        q = str(eval_input.get("question", ""))
        kps = eval_input.get("ground_truth_keypoints", {}) or {}
        vid = kps.get("video", []) if isinstance(kps, dict) else []
        txt = kps.get("text", []) if isinstance(kps, dict) else []
        ma = str(eval_input.get("model_answer", ""))
        # Use json.dumps for lists to keep them compact and valid
        vid_s = json.dumps(vid, ensure_ascii=False)
        txt_s = json.dumps(txt, ensure_ascii=False)
        return (
            f"question:\n{q}\n\n"
            f"ground_truth_keypoints:\n  video: {vid_s}\n  text: {txt_s}\n\n"
            f"model_answer:\n{ma}"
        )
    except Exception:
        # Fallback to raw JSON string
        return json.dumps(eval_input, ensure_ascii=False, indent=2)


def choose_best_fragment(reference: str, candidate: str) -> str:
    """Choose the sentence/fragment from `candidate` that best matches `reference`.

    - Splits `candidate` into sentence-like fragments (by punctuation and newlines).
    - Uses a lightweight SequenceMatcher ratio as a proxy for textual overlap.
    - Returns the fragment if it has higher ratio than the full candidate; otherwise returns full candidate.

    This is a small, conservative post-processing step used only for evaluation to favor
    concise correct fragments (helps short-answer models that provide a correct sentence).
    """
    try:
        ref = (reference or "").strip()
        cand = (candidate or "").strip()
        if not ref or not cand:
            return cand

        # Split into sentence-like fragments; keep punctuation as delimiter.
        frags = [s.strip() for s in __import__('re').split(r'(?<=[。.!?！？\n])\s+', cand) if s.strip()]
        if not frags:
            return cand

        # Score full candidate
        best_full_score = SequenceMatcher(None, ref, cand).ratio()
        best_frag = cand
        best_score = best_full_score

        for f in frags:
            score = SequenceMatcher(None, ref, f).ratio()
            if score > best_score:
                best_score = score
                best_frag = f

        return best_frag if best_frag else cand
    except Exception:
        return candidate
