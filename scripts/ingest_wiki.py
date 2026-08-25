"""P0 建库：把 quant-wiki 开源量化百科的概念词条灌入 QuantMind 知识库。

背景
----
QuantMind 端到端因子/策略挖掘链路中，「领域知识获取层」
（``quantmind.ai.knowledge_enrichment.enrich_idea``）的库内命中率长期偏低：
内置种子（``knowledge/seeds.py``）只有少量技术分析方法论，且知识库检索为
关键词子串匹配——没有高质量的概念/方法论资料，LLM 生成因子时缺乏
「标准定义 + 公式 + 实证参考」，只能联网抓取或自由发挥。

quant-wiki（https://github.com/LLMQuant/quant-wiki，CC BY-NC-SA 4.0，非商用）
提供 400+ 条中文量化词条（含 LaTeX 公式与参考文献）。本脚本将其
**因子/统计相关概念词条**离线解析、清洗后落库为 ``methodology`` 记录
（``source="wiki"`` 便于溯源与整体删除），幂等可重复执行。

收录范围（--sections concepts，默认）：
  - ``docs/basic/quant/``  全部 44 条（因子投资/回测/动量/贝塔/阿尔法/CAPM…）
  - ``docs/basic/finance/`` 因子/宏观因子相关白名单（多因子模型/波动性/市盈率…）
  - ``docs/basic/stat/`` 与 ``docs/basic/prob/`` 全部（回归/相关性/假设检验/IC 相关统计）
  （``index.md`` 与跨目录重复词条自动去重）

用法
----
    .\\venv\\Scripts\\python.exe scripts\\ingest_wiki.py \\
        --wiki-dir ..\\..\\_ref_wiki\\quant-wiki \\
        [--sections concepts] [--limit N] [--refresh] [--demo "动量因子"]

说明
----
- ``--refresh``：同 title + source="wiki" 的既有记录会被覆盖（默认幂等跳过）。
- ``--demo``：建库后对该查询词跑一次 ``KnowledgeStore.search`` 验证命中。
- 许可证注意：quant-wiki 为 CC BY-NC-SA 4.0（非商业性使用），本脚本产出仅供
  个人学习/研究；若 QuantMind 商用需替换内容源（知识层为可插拔设计）。
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from quantmind.knowledge.store import KnowledgeStore  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 落库时统一标记的来源（便于溯源/整体删除：DELETE FROM methodology WHERE source='wiki'）
SOURCE = "wiki"

#: basic/finance 下与因子研究强相关的词条（英文名白名单，用于过滤 150+ 金融术语）
FINANCE_WHITELIST = {
    "Multi-Factor Model", "Unlevered Beta", "Efficient Market Hypothesis",
    "Volatility", "Liquidity", "Price-to-Earnings Ratio (PdivE Ratio)",
    "Dividend", "Gross Profit Margin", "Interest Rate", "Inflation",
    "Economic Growth", "Exchange Rate", "Unemployment",
    "Gross Domestic Product (GDP)", "Rate of Change", "Annual Return",
    "Value Investing", "Margin", "Discount Rate",
}

#: 词条清洗：行首图片外链（含 gif/logo 横幅）
_IMG_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$", re.MULTILINE)

#: 词条清洗：页脚「## 关于LLMQuant」及其后全部内容
_FOOTER_RE = re.compile(r"\n#+\s*关于LLMQuant.*$", re.DOTALL)

#: 折叠块（mkdocs admonition）整体剥离：??? tip / ???+ example / !!! 等
_ADMON_RE = re.compile(r"^\s*\?{3}\+?\s*[^\n]*\n(?:.*\n)*?^\s*$", re.MULTILINE)

#: 标题行
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")


def clean_markdown(text: str) -> str:
    """清洗单条 wiki 词条：去图片外链、页脚、折叠块，压缩空行。"""
    t = text or ""
    t = _FOOTER_RE.sub("\n", t)
    t = _IMG_RE.sub("", t)
    t = _ADMON_RE.sub("", t)
    # 压缩 3+ 连续空行
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _first_paragraph(text: str) -> str:
    """取标题后的第一段非空纯文本（去掉 markdown 标记）。"""
    for para in text.split("\n\n"):
        s = re.sub(r"^#{1,6}\s*", "", para).strip()
        s = re.sub(r"[#*_`>]", "", s)
        if s:
            return s[:300]
    return ""


def _key_points(text: str) -> str:
    """提取「关键要点」列表（若有），否则返回空串。"""
    m = re.search(
        r"#+\s*关键要点\s*(.*?)(?=\n#+\s|\Z)", text, re.DOTALL
    )
    if not m:
        return ""
    points = re.findall(r"^\s*[-*]\s*(.+)$", m.group(1), re.MULTILINE)
    return "\n".join(points) if points else ""


def parse_concept(path: Path, section: str) -> Optional[Dict]:
    """解析单条词条 md → methodology dict（title/concept/summary/content/tags/meta）。"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = clean_markdown(raw)
    if not text:
        return None

    title = path.stem  # 形如「贝塔_Beta」/「动量投资_Momentum Investing」
    zh, _, en = title.partition("_")

    content = text
    summary = _key_points(text) or ""
    concept = _first_paragraph(text)

    tags = ["因子", "量化"] if section == "quant" else []
    if section == "finance":
        tags = ["金融", "因子"]
    elif section == "stat":
        tags = ["统计", "因子评估"]
    elif section == "prob":
        tags = ["概率", "因子评估"]
    if en:
        tags.append(en)

    meta = {
        "kind": "concept",
        "wiki_file": f"docs/basic/{section}/{path.name}",
        "source_type": "quant-wiki",
        "license": "CC BY-NC-SA 4.0 (non-commercial)",
        "zh_title": zh,
        "en_title": en or "",
    }
    return {
        "title": title,
        "concept": concept,
        "summary": summary,
        "content": content,
        "tags": tags,
        "meta": meta,
    }


def discover_concepts(wiki_dir: Path, limit: Optional[int] = None) -> List[Dict]:
    """扫描 quant-wiki 概念词条目录，返回解析后的 methodology dict 列表。"""
    basic = wiki_dir / "docs" / "basic"
    if not basic.is_dir():
        raise FileNotFoundError(f"未找到 quant-wiki docs/basic：{basic}")

    # quant / stat / prob：全收（排除 index.md）
    entries: List[Dict] = []
    for section in ("quant", "stat", "prob"):
        d = basic / section
        if not d.is_dir():
            continue
        for md in sorted(d.glob("*.md")):
            if md.name == "index.md":
                continue
            item = parse_concept(md, section)
            if item:
                entries.append(item)

    # finance：英文名白名单过滤
    fin_dir = basic / "finance"
    if fin_dir.is_dir():
        for md in sorted(fin_dir.glob("*.md")):
            if md.name == "index.md":
                continue
            en = md.stem.partition("_")[2]
            if en not in FINANCE_WHITELIST:
                continue
            item = parse_concept(md, "finance")
            if item:
                entries.append(item)

    # 跨目录重复 title 去重（如 stat/prob 各有「相关系数」）
    seen: set = set()
    dedup: List[Dict] = []
    for item in entries:
        if item["title"] in seen:
            continue
        seen.add(item["title"])
        dedup.append(item)

    if limit:
        dedup = dedup[:limit]
    return dedup


def _existing_titles(db_path: Path) -> set:
    """查 methodology 表中已存在的 (title) 集合（source='wiki'），用于幂等。"""
    if not db_path.exists():
        return set()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT title FROM methodology WHERE source=?", (SOURCE,)
        ).fetchall()
    return {r[0] for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="quant-wiki → QuantMind 知识库（P0 概念词条）")
    ap.add_argument("--wiki-dir", type=Path, required=True,
                    help="quant-wiki 仓库根目录（含 docs/basic）")
    ap.add_argument("--db", type=Path, default=None,
                    help="knowledge.db 路径（默认 quantmind/db/knowledge.db）")
    ap.add_argument("--limit", type=int, default=None,
                    help="仅处理前 N 条（调试用）")
    ap.add_argument("--refresh", action="store_true",
                    help="覆盖同 title 的既有 wiki 记录（默认幂等跳过）")
    ap.add_argument("--demo", type=str, default="动量因子",
                    help="建库后对该查询词跑一次检索 demo（空串跳过）")
    args = ap.parse_args()

    db_path = args.db or (PROJECT / "db" / "knowledge.db")
    wiki_dir = args.wiki_dir.resolve()

    entries = discover_concepts(wiki_dir, limit=args.limit)
    if not entries:
        print("没有解析到任何词条，检查 --wiki-dir")
        return 1

    store = KnowledgeStore(str(db_path))
    existing = _existing_titles(db_path)
    added = skipped = 0
    for item in entries:
        if item["title"] in existing and not args.refresh:
            skipped += 1
            continue
        store.ingest_methodology(
            title=item["title"],
            concept=item["concept"],
            summary=item["summary"],
            content=item["content"],
            source=SOURCE,
            tags=item["tags"],
            meta=item["meta"],
        )
        added += 1

    total = len(entries)
    print(f"共解析 {total} 条：新增 {added}，幂等跳过 {skipped}"
          f"{'（--refresh 覆盖）' if args.refresh else ''}")
    print(f"知识库：{db_path}")

    if args.demo:
        hits = store.search(args.demo, top_k=5, kind="methodology")
        print(f"\n检索 demo：'{args.demo}' top5")
        if not hits:
            print("  无命中！")
        for h in hits:
            print(f"  [{h['score']}] {h['metadata'].get('title', h['kb_id'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
