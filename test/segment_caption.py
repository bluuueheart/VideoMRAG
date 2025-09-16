from typing import List

from test_env_utils import image_to_base64

# 提供原 question_processing.py 中的段级多帧字幕生成逻辑（逐字搬迁，不改业务逻辑）
async def generate_segment_caption(frames: List[str], llm_cfg) -> str:
    """Generate an English multi-frame grounded caption.

    Pipeline:
      1) Per-frame objective sentence (no speculation) so model attends to every frame.
      2) Synthesize all unique sentences into ONE paragraph that integrates ALL visual evidence.
    """
    if not frames:
        return ""

    # --- Stage 1: per-frame objective description (English) ---
    async def _describe_single_frame(img_path: str, idx: int, total: int) -> str:
        b64 = image_to_base64(img_path)
        if not b64:
            return ""
        prompt = (
            f"Frame {idx+1}/{total}: Provide ONE objective English sentence (<=25 words) describing ONLY visible content: main scene, salient objects, on-screen text, charts, logos. "
            "NO speculation, NO inferred causes, NO symbolism. Forbidden words: maybe, might, seems, appear, appears, appear to, suggests, symbolize, symbolizes, represent, represents, possibly."
        )
        try:
            resp = await llm_cfg.cheap_model_func(
                prompt=prompt,
                system_prompt=(
                    "Return ONLY one concise objective English sentence. Do NOT speculate."
                ),
                images_base64=[b64],
                max_new_tokens=80,
                temperature=0.10,
                top_p=0.9
            )
        except Exception as e:
            print(f"[Caption][WARN] frame describe fail {img_path}: {e}")
            resp = ""

        # Be defensive: the model may return empty string or multiple blank lines.
        txt = ""
        try:
            if resp is None:
                resp = ""
            # Keep first non-empty line, fall back to empty string
            for ln in (resp or "").splitlines():
                if ln and ln.strip():
                    txt = ln.strip()
                    break
            if not txt:
                txt = (resp or "").strip()
        except Exception:
            txt = ""
        if txt.startswith('```'):
            txt = txt.strip('`')
        if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
            txt = txt[1:-1].strip()
        # Remove forbidden speculative words & trailing punctuation spaces
        forbidden = [
            "maybe", "might", "seems", "seem", "appears", "appear", "appear to",
            "suggests", "suggest", "symbolize", "symbolizes", "symbolise", "symbolises",
            "represent", "represents", "possibly", "perhaps"
        ]
        low = txt.lower()
        for w in forbidden:
            if w in low:
                # crude removal; could be improved by regex word boundaries
                txt = ' '.join([t for t in txt.split() if t.lower() != w])
                low = txt.lower()
        return txt.strip()

    frame_descs: List[str] = []
    for i, fp in enumerate(frames):
        try:
            d = await _describe_single_frame(fp, i, len(frames))
        except Exception as e:
            print(f"[Caption][ERR] single frame caption error: {e}")
            d = ""
        if d:
            frame_descs.append(d)
    # Deduplicate but preserve order
    seen = set(); uniq_descs = []
    for d in frame_descs:
        key = d.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq_descs.append(d)
    if not uniq_descs:
        return ""

    # --- Stage 2: Synthesis paragraph (English) ---
    # Build bullet evidence list
    joined = "\n".join(f"- {s}" for s in uniq_descs)
    # NOTE: Previous version BUG: the bullet list (joined) was never injected into the prompt, causing the model
    # to rely only on a static instruction block (with an example about Excel / charts / EV specs), leading to
    # hallucinated / cross-video contamination. We now explicitly include ONLY the extracted per-frame sentences
    # as the sole evidence, and we REMOVED the concrete example to avoid lexical anchoring.
    synth_prompt = f"""
### **Visual Evidence Synthesis**

**Goal:**
Your goal is to **distill** the provided facts into a **single, succinct, and natural-sounding English paragraph**. The final output should be a refined summary that seamlessly integrates all key information, not just a list of sentences strung together.

**Evidence:**
You will receive a set of objective, unordered facts from the keyframes of a short video clip.
`{joined}`

**Instructions & Constraints:**

1.  **Synthesize, Don't Just List:** Analyze all facts to understand the complete picture. Weave the non-redundant details into a single, coherent paragraph where sentences flow logically.
2.  **Be Precise and Grounded:** Every detail in your output must originate from the provided evidence. Preserve all specific entities, numbers, on-screen text, and UI labels **exactly** as they appear.
3.  **No Invention or Commentary:**
    * Do **NOT** add information that is not present in the evidence.
    * Do **NOT** use meta-phrases like "The image shows," or "In the frame...".
    * Maintain a strictly neutral and factual tone.

**Output:**
A single, clean paragraph of text (no numbering, no quotes, no markdown)."""
    try:
        resp2 = await llm_cfg.cheap_model_func(
            prompt=synth_prompt,
            system_prompt="Produce ONE objective English paragraph. Do not speculate.",
            images_base64=None,
            max_new_tokens=400,
            temperature=0.2,
            top_p=0.9
        )
    except Exception as e:
        print(f"[Caption][ERR] synthesis failed: {e}")
        resp2 = ""
    para = (resp2 or "").strip()
    if para.startswith('```'):
        parts = [ln for ln in para.splitlines() if not ln.strip().startswith('```')]
        para = " ".join(parts).strip()
    if (para.startswith('"') and para.endswith('"')) or (para.startswith("'") and para.endswith("'")):
        para = para[1:-1].strip()
    # Remove speculative words again defensively
    for w in [
        "maybe", "might", "seems", "seem", "appears", "appear", "appear to",
        "suggests", "suggest", "symbolize", "symbolizes", "symbolise", "symbolises",
        "represent", "represents", "possibly", "perhaps"
    ]:
        if w in para.lower():
            tokens = []
            for t in para.split():
                if t.lower() != w:
                    tokens.append(t)
            para = " ".join(tokens)
    if len(para) < 20:  # fallback if model produced too little
        para = " ".join(uniq_descs)
    return para.strip()
