"""LLM 结构化抽取：从研报全文中提取因子实证信息。

背景
----
当前 ingest_reports.py 用启发式规则抽取 IC/IR、样本期等信息，效果有限。
本脚本用 LLM 从研报全文中提取结构化信息：
- 因子名称（如"动量因子"、"波动率因子"）
- IC/IR（信息系数/信息比率）
- 样本期（如"2010-2020"）
- 适用市场（如"A股"、"期货"）
- 因子构造方法（一句话描述）
- 因子有效性结论（如"显著"、"不显著"）

用法
----
    .\\venv\\Scripts\\python.exe scripts\\enrich_reports_llm.py \\
        [--limit 10] [--batch-size 5] [--refresh]

参数
----
--limit       最多处理 N 条研报（调试用）
--batch-size  每批处理 N 条（避免 API 限流，默认 5）
--refresh     覆盖已处理的记录（默认跳过）

说明
----
- 依赖 .env 中的 QM_LLM_* 配置
- 幂等：已处理的记录（meta.enriched=true）默认跳过
- 失败降级：LLM 调用失败时保留原 meta
- 成本估算：约 0.01-0.05 元/篇（取决于模型和内容长度）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from quantmind.ai.provider import build_provider
from quantmind.knowledge.store import KnowledgeStore

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """你是一个量化投研专家。请从以下研报内容中提取结构化信息。

研报标题：{title}

研报内容（前 3000 字符）：
{content}

请提取以下信息（JSON 格式）：
{{
  "factors": ["因子名称列表，如：动量因子、波动率因子"],
  "ic_ir": "IC/IR 数值，如：IC=0.05, IR=1.2",
  "sample_period": "样本期，如：2010-2020",
  "market": "适用市场，如：A股、期货、港股",
  "construction_method": "因子构造方法，一句话描述",
  "conclusion": "因子有效性结论，如：显著、不显著、条件显著"
}}

要求：
1. 如果某项信息未在研报中明确提及，填 null 或空字符串
2. factors 是列表，其他字段是字符串
3. 只提取研报中明确提到的信息，不要推测
4. 返回纯 JSON，不要 markdown 代码块

请直接返回 JSON："""


async def extract_with_llm(title: str, content: str, provider) -> Optional[Dict]:
    """用 LLM 从研报内容中提取结构化信息。"""
    # 截取前 3000 字符（避免超长）
    content_preview = content[:3000]
    prompt = EXTRACT_PROMPT.format(title=title, content=content_preview)
    
    try:
        response = await provider.chat(
            "你是量化投研专家，擅长从研报中提取因子实证信息。",
            prompt
        )
        
        # 解析 JSON（处理可能的 markdown 代码块）
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        
        extracted = json.loads(json_str)
        return extracted
    except Exception as e:
        print(f"    ✗ LLM 提取失败：{e}")
        return None


def get_unenriched_reports(db_path: Path, limit: Optional[int] = None) -> List[Dict]:
    """获取未处理的研报记录（meta.enriched != true）。"""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT kb_id, title, content, meta
            FROM methodology
            WHERE source = 'wiki-report'
        """
        rows = conn.execute(query).fetchall()
    
    reports = []
    for row in rows:
        meta = json.loads(row["meta"]) if row["meta"] else {}
        if meta.get("enriched"):
            continue
        reports.append({
            "kb_id": row["kb_id"],
            "title": row["title"],
            "content": row["content"],
            "meta": meta,
        })
        if limit and len(reports) >= limit:
            break
    
    return reports


def update_meta(db_path: Path, kb_id: str, meta: Dict) -> None:
    """更新 methodology 表的 meta 字段。"""
    meta_json = json.dumps(meta, ensure_ascii=False)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE methodology SET meta = ? WHERE kb_id = ?",
            (meta_json, kb_id)
        )


async def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 结构化抽取研报信息")
    ap.add_argument("--db", type=Path, default=None,
                    help="knowledge.db 路径（默认 quantmind/db/knowledge.db）")
    ap.add_argument("--limit", type=int, default=None,
                    help="最多处理 N 条研报（调试用）")
    ap.add_argument("--batch-size", type=int, default=5,
                    help="每批处理 N 条（避免 API 限流，默认 5）")
    ap.add_argument("--refresh", action="store_true",
                    help="覆盖已处理的记录（默认跳过）")
    args = ap.parse_args()
    
    db_path = args.db or (PROJECT / "db" / "knowledge.db")
    
    # 加载 .env
    env_path = PROJECT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)
    
    # 构建 provider
    provider = build_provider(
        os.environ.get("QM_LLM_PROVIDER", "openai"),
        os.environ.get("QM_LLM_API_KEY", ""),
        os.environ.get("QM_LLM_BASE_URL", ""),
        os.environ.get("QM_LLM_MODEL", ""),
    )
    
    if not os.environ.get("QM_LLM_API_KEY"):
        print("错误：未配置 QM_LLM_API_KEY，请在 .env 中配置")
        return 1
    
    # 获取未处理的研报
    reports = get_unenriched_reports(db_path, limit=args.limit)
    if not reports:
        print("没有需要处理的研报（全部已 enriched）")
        return 0
    
    print(f"待处理：{len(reports)} 篇研报")
    print(f"批次大小：{args.batch_size}")
    print("=" * 60)
    
    # 批量处理
    enriched = 0
    failed = 0
    
    for i in range(0, len(reports), args.batch_size):
        batch = reports[i:i + args.batch_size]
        print(f"\n批次 {i // args.batch_size + 1}：{len(batch)} 篇")
        
        for report in batch:
            print(f"\n  [{enriched + failed + 1}/{len(reports)}] {report['title'][:50]}...")
            
            # LLM 提取
            extracted = await extract_with_llm(
                report["title"],
                report["content"],
                provider
            )
            
            if extracted is None:
                failed += 1
                continue
            
            # 更新 meta
            meta = report["meta"]
            meta.update({
                "factors": extracted.get("factors", []),
                "ic_ir": extracted.get("ic_ir", ""),
                "sample_period": extracted.get("sample_period", ""),
                "market": extracted.get("market", ""),
                "construction_method": extracted.get("construction_method", ""),
                "conclusion": extracted.get("conclusion", ""),
                "enriched": True,
            })
            
            update_meta(db_path, report["kb_id"], meta)
            enriched += 1
            
            print(f"    ✓ 因子：{extracted.get('factors', [])}")
            print(f"      IC/IR：{extracted.get('ic_ir', '未识别')}")
            print(f"      样本期：{extracted.get('sample_period', '未识别')}")
            print(f"      市场：{extracted.get('market', '未识别')}")
            print(f"      结论：{extracted.get('conclusion', '未识别')}")
        
        # 批次间延迟（避免 API 限流）
        if i + args.batch_size < len(reports):
            await asyncio.sleep(1)
    
    print("\n" + "=" * 60)
    print(f"总计：enriched={enriched}, failed={failed}")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
