import json
import os

from test_env_utils import extract_final_answer

# 拆分：负责 prompt 构建与调用 LLM（从 question_processing 中抽离，原逻辑逐字保留）
async def build_and_call_llm(query, all_segment_data, llm_cfg, base_mode: bool = False, images_base64=None):
    print("\n--- [Step 3] Constructing multimodal prompt ---")
    llm_input_chunks = []
    all_images_b64 = []
    image_counter = 0
    # 当 base_mode=True 时，发送图片（25 张），否则使用 caption（原逻辑）
    images_disabled = not bool(base_mode)
    # --- 新增: token 统计辅助 ---
    def _approx_token_count(text: str) -> int:
        try:
            import tiktoken  # 若存在更精确  # type: ignore
            enc = tiktoken.get_encoding('cl100k_base')
            return len(enc.encode(text))
        except Exception:
            # 简单启发: 按空格分 + 标点近似
            if not text:
                return 0
            return max(1, len(text.strip().split()))

    total_tokens_text = 0

    # 按需关闭摘要：最终输入仅使用重采样帧 + 原始字幕 + OCR/DET
    use_diff_flag = False
    per_clip_logs = []

    # 保留上游已经排好（含邻居 prev-base-next）顺序，不再按 id 重新排序
    # caption 统计
    _cap_total_chars = 0
    _cap_non_empty = 0
    _cap_clips = 0

    for data in all_segment_data:
        clip_id = data['id']
        chunk_lines = [f"[Chunk: {clip_id}]"]
        # 段级 caption（普通模式使用，base_mode 下不使用 caption）
        seg_cap = '' if base_mode else (data.get('segment_caption') or '').strip()
        if seg_cap:
            chunk_lines.append(f"Caption: \"{seg_cap}\"")
        # caption 统计累积
        _cap_clips += 1
        if seg_cap:
            _cap_non_empty += 1
            _cap_total_chars += len(seg_cap)
        # 1. 视觉差异摘要 (若可用)
        diff_text = data.get('diff_summary') if use_diff_flag else None
        diff_text = (diff_text or '').strip()
        # 2. OCR / DET
        ocr_text = (data.get('ocr_text') or '').strip()
        det_text = (data.get('det_text') or '').strip()
        # 3. 原字幕（始终使用完整字幕，不截断）
        raw_sub = (data.get('transcript') or '').strip()
        subtitle_snippet = raw_sub

        # 组装文本块（顺序: caption -> diff -> OCR -> DET -> Subtitles）
        if diff_text:
            chunk_lines.append(f"RefinedVisualSummary: \"{diff_text}\"")
        if ocr_text:
            chunk_lines.append(f"On-Screen Text: \"{ocr_text}\"")
        if det_text:
            chunk_lines.append(f"Detected Objects: \"{det_text}\"")
        chunk_lines.append(f"Subtitles: \"{subtitle_snippet}\"")

        # 统计 token
        diff_tokens = _approx_token_count(diff_text) if diff_text else 0
        ocr_tokens = _approx_token_count(ocr_text) if ocr_text else 0
        det_tokens = _approx_token_count(det_text) if det_text else 0
        sub_tokens = _approx_token_count(subtitle_snippet)
        clip_total = diff_tokens + ocr_tokens + det_tokens + sub_tokens
        total_tokens_text += clip_total
        per_clip_logs.append(
            f"[PromptBuild][{clip_id}] diff={diff_tokens} ocr={ocr_tokens} det={det_tokens} sub={sub_tokens} total={clip_total}"
        )

        # Keyframes
        frames_with_ts = data.get("frames_with_ts", [])
        if frames_with_ts:
            if images_disabled:
                chunk_lines.append(f"Keyframes: {len(frames_with_ts)} frames (images disabled)")
            else:
                chunk_lines.append(f"Keyframes: {len(frames_with_ts)} frames (images included)")
        else:
            if images_disabled:
                chunk_lines.append("Keyframes: 0 frames (images disabled)")
            else:
                chunk_lines.append("Keyframes: 0 frames (images included)")

        llm_input_chunks.append("\n".join(chunk_lines))

    # 输出 token 总览日志
    print("[PromptBuild] Per-clip token usage (approx):")
    for line in per_clip_logs:
        print(line)
    print(f"[PromptBuild] Aggregate text tokens (approx): {total_tokens_text}")
    if images_disabled:
        print(f"[PromptBuild] Images disabled -> 0 images will be sent (captions used instead).")
    else:
        total_imgs = len(images_base64) if images_base64 else 0
        print(f"[PromptBuild] Images enabled -> {total_imgs} images will be sent (base_mode)")
        # 如果 images_base64 是带元数据的 dict 列表，输出简短预览
        try:
            if images_base64 and isinstance(images_base64, list) and isinstance(images_base64[0], dict):
                preview = []
                for it in images_base64[:5]:
                    preview.append(f"{it.get('segment_id')}#f{it.get('frame_index')}@{it.get('timestamp')}")
                print(f"[PromptBuild] Images preview (first 5): {preview}")
        except Exception:
            pass
    # 输出 caption 汇总日志
    try:
        _avg_cap = (_cap_total_chars // _cap_non_empty) if _cap_non_empty else 0
        print(f"[Answer-Caption] clips={_cap_clips} non_empty={_cap_non_empty} total_chars={_cap_total_chars} avg_non_empty_chars={_avg_cap}")
    except Exception:
        pass

    context_str = "\n\n".join(llm_input_chunks)
    
    final_prompt_template = """
### **Multimodal Evidence Synthesis for Question Answering**

**Role:**
You are a specialized AI assistant designed to answer questions by extracting precise information from video evidence.

**Objective:**
Your sole objective is to provide a **direct and focused** answer to the user's question, using only the pieces of evidence that are **strictly necessary**.

**Context Description:**
You will receive keyframes and subtitles from 5 video segments. Your task is to sift through this evidence to find the answer to a specific question. Much of the provided information may be irrelevant.

**Step-by-Step Task Instructions:**

1.  **Filter for Relevance:** First, scan all provided evidence (keyframes and subtitles) with the singular purpose of identifying which segments, if any, contain information that is pertinent to the user's question. **Immediately dismiss any segments that do not directly contribute to the answer.**
2.  **Extract Essential Details:** From the **relevant segments only**, extract the specific facts—whether textual or visual—that are absolutely required to construct the answer. Disregard all other information within these segments.
3.  **Construct a Direct Answer:** Integrate the essential details into the most concise and direct answer possible. The response must be a straightforward answer to the question, without background, context, or tangential facts.

**Critical Constraints:**

* **Principle of Minimum Information:** Your response **MUST** adhere to this principle. Provide only the information required to fully and accurately answer the question. If a detail is interesting but not essential, it **MUST** be excluded.
* **Absolute Grounding in Evidence:** Your answer **MUST** be derived **exclusively** from the information contained within the provided subtitles and keyframes. This is your only source of truth.
* **No External Knowledge:** You **MUST NOT** use any pre-existing knowledge, make assumptions, or infer information that is not explicitly presented in the provided evidence.
* **Insufficiency Clause:** If, after filtering, there is not enough evidence to answer the question, you **MUST** state: "The provided information is insufficient to answer the question."
* **Output Format:** The final output **MUST** be a single, valid JSON object. Do not include any additional text, explanations, or markdown formatting outside of the specified JSON structure.

**Input:**
```
User Question: {user_question}

Evidence: {context_str}
```

**Output Specification:**
Your response must be a single JSON object in the following format:

```json
{{
  "answer": "..."
}}
```"""
    final_prompt = final_prompt_template.format(user_question=query, context_str=context_str)

    print("\n--- [Step 4] Sending prompt to LLM ---")
    print(f"Prompt text length: {len(final_prompt)}")
    print(f"Number of images: {len(images_base64) if images_base64 else 0}")
    print("----------------------------------\n")

    param_response_type = 'The final output MUST be a single, valid JSON object. Do not include any additional text, explanations, apologies, or markdown formatting outside of the JSON structure.'

    # 兼容性：有些后端期望 images_base64 为 list[str]，有些可接受 list[dict]
    images_payload = None
    if not images_disabled and images_base64:
        if isinstance(images_base64, list) and isinstance(images_base64[0], dict):
            # 从 dict list 中提取出 base64 字符串字段（'image_base64'）并保证为纯 base64 字符串
            extracted = []
            for it in images_base64:
                try:
                    b = it.get('image_base64') if isinstance(it, dict) else it
                    if not b:
                        continue
                    # 如果包含 data:image 前缀，保留完整字符串（后端会处理），否则使用原始 b64
                    extracted.append(b)
                except Exception:
                    continue
            images_payload = extracted if extracted else None
        else:
            images_payload = images_base64
        # If llm_cfg requests VLM acceleration, surface a log and allow backend to optimize accordingly
        try:
            if hasattr(llm_cfg, 'vlm_accel') and getattr(llm_cfg, 'vlm_accel'):
                print('[LLM] VLM acceleration requested (llm_cfg.vlm_accel=True)')
                # Optionally: tune generation params when VLM accel requested
                # This is intentionally minimal: downstream clients may inspect llm_cfg.vlm_accel
        except Exception:
            pass
    response = await llm_cfg.cheap_model_func(
        prompt=final_prompt,
        system_prompt=param_response_type,
        images_base64=images_payload,
        max_new_tokens=512,
        temperature=0.25,
        top_p=0.9
    )

    print("\n--- [Step 5] Processing LLM response ---")
    answer_obj = None
    try:
        start_index = response.find('{')
        end_index = response.rfind('}') + 1
        if start_index != -1 and end_index != -1:
            json_str = response[start_index:end_index]
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer_obj = {"answer": str(parsed.get("answer", ""))}
            else:
                answer_obj = {"answer": str(response).strip()}
        else:
            answer_obj = {"answer": str(response).strip()}
    except Exception:
        answer_obj = {"answer": str(response or "").strip()}

    # --- 新增：统一抽取最终干净答案 ---
    answer_obj["answer"] = extract_final_answer(answer_obj.get("answer", ""))

    # --- 新增: 结果质量快速判别 & 一次性重写尝试 (避免 caption 罗列) ---
    try:
        raw_ans = answer_obj.get("answer", "")
        def _looks_like_caption_dump(txt: str) -> bool:
            if not txt: return False
            # 规则: 多个句子以 "The image"/"A person" 等反复开头, 或 纯对象逗号序列 > 8 个
            import re
            lower = txt.lower()
            repetitive = len(re.findall(r"\b(the image|a person|a man|a woman|there is)\b", lower)) >= 3
            comma_list = (lower.count(",") >= 8 and not any(k in lower for k in ["because", "therefore", "thus", "所以"]))
            no_question_terms = not any(qk in lower for qk in ["compare", "difference", "趋势", "变化", "increase", "decrease", "稳定", "stability"])
            return (repetitive or comma_list) and no_question_terms
        if _looks_like_caption_dump(raw_ans):
            print("[AnswerCheck] Detected caption-like dump; triggering concise analytic rewrite.")
            rewrite_prompt = f"You are revising an answer that mistakenly described visual frames instead of answering the question.\nQuestion: {query}\nFaultyAnswer: {raw_ans}\nRewrite a concise, evidence-grounded answer (or state insufficiency). Output only JSON {{\"answer\": \"...\"}}."
            try:
                rewrite_resp = await llm_cfg.cheap_model_func(prompt=rewrite_prompt, system_prompt="Return JSON only.", images_base64=None, max_new_tokens=256, temperature=0.2, top_p=0.9)
                si = rewrite_resp.find('{'); ei = rewrite_resp.rfind('}') + 1
                if si != -1 and ei != -1:
                    parsed2 = json.loads(rewrite_resp[si:ei])
                    if isinstance(parsed2, dict) and parsed2.get('answer'):
                        answer_obj['answer'] = extract_final_answer(parsed2['answer'])
                        answer_obj['rewrite'] = True
            except Exception as _rew_err:
                print(f"[AnswerCheck][WARN] rewrite failed: {_rew_err}")
    except Exception:
        pass
    # --- 新增: 占位符/过短答案自动重生成 ---
    try:
        raw_ans = (answer_obj.get("answer", "") or "").strip()
        # 可配置的最小答案长度与重试次数
        min_len = int(os.environ.get("FINAL_MIN_ANSWER_CHARS", "6"))
        attempts = int(os.environ.get("FINAL_RETRY_ATTEMPTS", "1"))
        # 视为占位符或无效的简单规则
        placeholders = {"", "...", "n/a", "no answer", "unknown"}

        def _is_placeholder(s: str) -> bool:
            if not s:
                return True
            ls = s.strip().lower()
            if ls in placeholders:
                return True
            if len(s) < min_len:
                return True
            return False

        if _is_placeholder(raw_ans):
            print(f"[AnswerCheck] Detected placeholder/too-short answer ('{raw_ans}'), attempting up to {attempts} regeneration(s)")
            regen_prompt = (
                "You are given a question and evidence. Produce a single JSON object like {\"answer\": \"...\"} with a concise, evidence-grounded answer. "
                f"Question: {query}\nEvidence: {context_str}\nFaultyAnswer: {raw_ans}"
            )
            for i in range(attempts):
                try:
                    resp = await llm_cfg.cheap_model_func(
                        prompt=regen_prompt,
                        system_prompt='Return JSON only.',
                        images_base64=None,
                        max_new_tokens=256,
                        temperature=0.2,
                        top_p=0.9,
                    )
                    si = resp.find('{')
                    ei = resp.rfind('}') + 1
                    candidate = None
                    if si != -1 and ei != -1:
                        try:
                            parsed2 = json.loads(resp[si:ei])
                            if isinstance(parsed2, dict) and parsed2.get('answer'):
                                candidate = extract_final_answer(parsed2.get('answer'))
                        except Exception:
                            candidate = extract_final_answer(resp[si:ei])
                    else:
                        candidate = extract_final_answer(resp)

                    if candidate and not _is_placeholder(candidate):
                        answer_obj['answer'] = candidate
                        answer_obj['regenerated'] = True
                        print(f"[AnswerCheck] Regeneration succeeded on attempt {i+1}")
                        break
                    else:
                        print(f"[AnswerCheck] Regeneration attempt {i+1} produced invalid/placeholder result: '{candidate}'")
                except Exception as e:
                    print(f"[AnswerCheck] Regeneration attempt {i+1} failed: {e}")
    except Exception:
        pass

    return answer_obj
