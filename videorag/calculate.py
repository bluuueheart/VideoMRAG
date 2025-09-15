#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按文件夹聚合 /root/autodl-tmp/Result/<子文件夹> 下所有 JSON 文件的结果，计算：
- recall(video)   = sum(covered_video_keypoints) / sum(videokeypoints or gt_video_n)
- recall(text)    = sum(covered_text_keypoints) / sum(textkeypoints or gt_text_n)
- recall(all)     = (sum(covered_video_keypoints)+sum(covered_text_keypoints)) / (sum(video)+sum(text))
- Precision       = (sum(covered_video_keypoints)+sum(covered_text_keypoints)) / sum(total_claimed_keypoints)
- F1-Score        = 2 * Precision * recall(all) / (Precision + recall(all))
输出每个子文件夹的：recall(text) recall(video) recall(all) Precision F1-Score 平均st_cosine 平均likert_score 平均rouge_l

使用：
    python compute_result_metrics.py --base <RESULT_BASE> --out metrics.csv
    # 默认会从统一配置 `videorag._config` 的 `get_root_prefix()` 派生（或使用环境变量 `RESULT_BASE` 覆盖）

说明：
- JSON 文件为列表或单条记录，字段兼容：
  gt_video_n 或 videokeypoints；gt_text_n 或 textkeypoints；rouge_l_f 或 rouge_l；bertscore_f1 或 bertscore
- 分母为 0 时，该比值记为 0.0
- 平均值忽略 None/空值；若全为空则为空字符串
"""

import argparse
import os
import json
import math
from typing import Any, Dict, Iterable, List, Optional

# Centralized path configuration: import from videorag._config
try:
    from ._config import get_root_prefix, OUTPUT_BASE_DIR_DEFAULT
except Exception:
    from videorag._config import get_root_prefix, OUTPUT_BASE_DIR_DEFAULT

Number = Optional[float]

def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    try:
        xf = float(str(x).strip())
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf
    except Exception:
        return None

def safe_div(numer: float, denom: float) -> float:
    # If denominator is zero, return None to avoid confusing missing with 0.0
    if denom in (0, 0.0):
        return None
    return (numer / denom)

def mean_ignore_none(values: Iterable[Number]) -> Optional[float]:
    s = 0.0
    c = 0
    for v in values:
        if v is None:
            continue
        s += v
        c += 1
    return (s / c) if c else None

def read_json_file(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ("items", "data", "results"):
            v = data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [data]
    return []

def collect_items_in_folder(folder: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith('.json'):
                continue
            path = os.path.join(root, fn)
            try:
                items.extend(read_json_file(path))
            except Exception as e:
                print(f"[WARN] 跳过无法解析的文件: {path}: {e}")
    return items

def compute_folder_metrics(items: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    # Prefer LLM-reported coverage when present (llm_covered_*),
    # otherwise fall back to locally-computed 'covered_*'.
    # This keeps aggregation consistent when some result files contain
    # both heuristic and LLM-evaluated coverage counts.
    def _cov_value(it, llm_key: str, raw_key: str):
        # Prefer llm-reported coverage; if absent, fall back to raw covered value;
        # if both missing return None to indicate missing data (not 0)
        v_llm = to_float(it.get(llm_key))
        if v_llm is not None:
            return v_llm
        v_raw = to_float(it.get(raw_key))
        if v_raw is not None:
            return v_raw
        return None

    # Sum only available coverage values; keep None when all are missing so ratios can be None
    sum_cov_v = 0.0
    sum_cov_t = 0.0
    any_cov_v = False
    any_cov_t = False
    for it in items:
        cv = _cov_value(it, 'llm_covered_video_keypoints', 'covered_video_keypoints')
        if cv is not None:
            sum_cov_v += cv
            any_cov_v = True
        ct = _cov_value(it, 'llm_covered_text_keypoints', 'covered_text_keypoints')
        if ct is not None:
            sum_cov_t += ct
            any_cov_t = True

    def get_gt_v(it):
        v = to_float(it.get('gt_video_n'))
        return v if v is not None else to_float(it.get('videokeypoints'))
    def get_gt_t(it):
        v = to_float(it.get('gt_text_n'))
        return v if v is not None else to_float(it.get('textkeypoints'))

    sum_gt_v = 0.0
    sum_gt_t = 0.0
    sum_claimed = 0.0
    any_gt_v = False
    any_gt_t = False
    any_claimed = False
    for it in items:
        gv = get_gt_v(it)
        if gv is not None:
            sum_gt_v += gv
            any_gt_v = True
        gt = get_gt_t(it)
        if gt is not None:
            sum_gt_t += gt
            any_gt_t = True
        tc = to_float(it.get('total_claimed_keypoints'))
        if tc is not None:
            sum_claimed += tc
            any_claimed = True

    # Collect metrics strictly from LLM-returned fields only.
    # - likert: use 'likert_score' if present (0 is a valid value)
    # - rouge: use 'rouge_l_f' or 'rouge_l' (computed by dependency in evaluate)
    # - st_cosine: use 'st_cosine_score' (returned by dedicated model in evaluate)
    likerts: List[float] = []
    rouges: List[Optional[float]] = []
    st_cosines: List[float] = []
    # collect likert sub-scores if present
    factual_cov: List[Optional[float]] = []
    visual_usage: List[Optional[float]] = []
    ling_prec: List[Optional[float]] = []
    for it in items:
        # likert from LLM
        v_likert = to_float(it.get('likert_score'))
        if v_likert is not None:
            likerts.append(v_likert)

        # rouge from evaluate's dependency
        rv = to_float(it.get('rouge_l_f'))
        if rv is None:
            rv = to_float(it.get('rouge_l'))
        rouges.append(rv)

        # st_cosine returned by evaluate (sentence-transformers or None)
        sv = to_float(it.get('st_cosine_score'))
        st_cosines.append(sv)
        # likert_subscores may be a dict with integer subscores
        subs = it.get('likert_subscores')
        if isinstance(subs, dict):
            factual_cov.append(to_float(subs.get('factual_coverage')))
            visual_usage.append(to_float(subs.get('visual_detail_usage')))
            ling_prec.append(to_float(subs.get('linguistic_precision')))

    # Compute ratios; if denominator missing or zero, safe_div returns None
    recall_video = safe_div(sum_cov_v, sum_gt_v) if any_cov_v and any_gt_v else None
    recall_text = safe_div(sum_cov_t, sum_gt_t) if any_cov_t and any_gt_t else None
    recall_all = safe_div(sum_cov_v + sum_cov_t, (sum_gt_v + sum_gt_t)) if (any_cov_v or any_cov_t) and (any_gt_v or any_gt_t) else None
    precision = safe_div(sum_cov_v + sum_cov_t, sum_claimed) if (any_cov_v or any_cov_t) and any_claimed else None
    # f1 should be None only if precision or recall_all is missing; if both present but sum==0, f1==0.0
    if precision is None or recall_all is None:
        f1 = None
    else:
        if (precision + recall_all) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall_all / (precision + recall_all)

    # avg_st_cosine: use only st_cosine_score returned by evaluate (no fallback)
    avg_st = mean_ignore_none(st_cosines)

    return {
        'recall_text': recall_text,
        'recall_video': recall_video,
        'recall_all': recall_all,
        'precision': precision,
        'f1': f1,
    'avg_st_cosine': avg_st,
    'avg_likert': mean_ignore_none(likerts),
    'avg_factual_coverage': mean_ignore_none(factual_cov),
    'avg_visual_detail_usage': mean_ignore_none(visual_usage),
    'avg_linguistic_precision': mean_ignore_none(ling_prec),
        'avg_rouge_l': mean_ignore_none(rouges),
        'count': float(len(items)),
    }

def format_float(v: Optional[float], ndigits: int = 4) -> str:
    if v is None:
        return 'null'
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return '-'

def list_immediate_subdirs(path: str) -> List[str]:
    try:
        with os.scandir(path) as it:
            subs = [e.path for e in it if e.is_dir()]
    except FileNotFoundError:
        print(f"[ERROR] 基础目录不存在: {path}")
        return []
    return sorted(subs)

def main():
    parser = argparse.ArgumentParser(description="聚合 Result 子文件夹 JSON 计算指标")
    # Derive default base from centralized config. Prefer explicit env RESULT_BASE if set.
    default_base = os.environ.get('RESULT_BASE') or os.path.join(get_root_prefix(), 'lx', 'Result')
    parser.add_argument('--base', type=str, default=default_base, help='基础目录，包含若干子文件夹')
    parser.add_argument('--out', type=str, default=None, help='输出 CSV 路径（默认写到 <base>/folder_metrics.csv）')
    parser.add_argument('--debug', action='store_true', help='打印每个子文件夹的原始 per-item 指标以便调试')
    args = parser.parse_args()

    base = args.base
    out_csv = args.out or os.path.join(base, 'folder_metrics.csv')

    subdirs = list_immediate_subdirs(base)
    header = [
        'folder',
        'recall_text', 'recall_video', 'recall_all', 'precision', 'f1',
    'avg_st_cosine', 'avg_likert',
    'avg_factual_coverage', 'avg_visual_detail_usage', 'avg_linguistic_precision',
    'avg_rouge_l',
        'n_items',
    ]
    rows: List[List[str]] = []

    for sub in subdirs:
        items = collect_items_in_folder(sub)
        metrics = compute_folder_metrics(items)
        if args.debug:
            # Debug: show per-item metric lists to trace aggregation source
            lik_list = [to_float(it.get('likert_score')) for it in items]
            rouge_list = [to_float(it.get('rouge_l_f') or it.get('rouge_l')) for it in items]
            st_list = [to_float(it.get('st_cosine_score')) for it in items]
            print(f"[DEBUG] {os.path.basename(sub)}: likert={lik_list}, rouge={rouge_list}, st={st_list}")
        row = [
            os.path.basename(sub.rstrip(os.sep)),
            format_float(metrics['recall_text'], 4),
            format_float(metrics['recall_video'], 4),
            format_float(metrics['recall_all'], 4),
            format_float(metrics['precision'], 4),
            format_float(metrics['f1'], 4),
            format_float(metrics['avg_st_cosine'], 4),
            format_float(metrics['avg_likert'], 4),
            format_float(metrics.get('avg_factual_coverage'), 4),
            format_float(metrics.get('avg_visual_detail_usage'), 4),
            format_float(metrics.get('avg_linguistic_precision'), 4),
            format_float(metrics['avg_rouge_l'], 4),
            str(int(metrics.get('count') or 0)),
        ]
        rows.append(row)

    # 控制台和文件美化输出
    col_widths = [max(len(str(x)) for x in [h]+[r[i] for r in rows]) for i, h in enumerate(header)]
    def fmt_row(row):
        return ' | '.join([
            str(row[0]).ljust(col_widths[0])
        ] + [str(row[i]).rjust(col_widths[i]) for i in range(1, len(row))])

    table_lines = []
    table_lines.append(fmt_row(header))
    table_lines.append('-+-'.join(['-'*w for w in col_widths]))
    for r in rows:
        table_lines.append(fmt_row(r))

    print('\n' + '\n'.join(table_lines) + '\n')

    # 写出到 CSV 文件（表格文本格式，非逗号分隔）
    try:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, 'w', encoding='utf-8') as f:
            for line in table_lines:
                f.write(line + '\n')
        print(f"[INFO] 已写出: {out_csv}")
    except Exception as e:
        print(f"[WARN] 写出 CSV 失败: {e}")

if __name__ == '__main__':
    main()