"""P1 研报 PDF 管道：解析研报目录 → 下载 PDF → 解析 → 入库。

背景
----
quant-wiki 的「研报精选」目录收录了 7 大券商的多因子系列研报（中信/华泰/国盛/广发/海通等），
每篇研报都包含因子构造方法、IC/IR 实证、样本期、适用市场等关键信息。这些信息对因子挖掘
极具价值，但当前以 PDF 外链形式存在，未入库。

本脚本实现：
1. 解析研报目录 md（如中信多因子系列的 index.md），提取标题和 PDF URL
2. 批量下载 PDF 到本地（data_cache/reports/）
3. 用 PyMuPDF 解析 PDF，抽取全文文本
4. 结构化抽取：因子名称、构造方法、IC/IR、样本期、适用市场（启发式规则）
5. 入库为 methodology 记录（source="wiki-report"）

用法
----
    .\\venv\\Scripts\\python.exe scripts\\ingest_reports.py \\
        --wiki-dir ..\\..\\_ref_wiki\\quant-wiki \\
        --series "中信-多因子系列" \\
        [--limit 3] [--refresh] [--demo "因子衰减"]

说明
----
- ``--series``：研报系列名称（如"中信-多因子系列"），不传则处理全部系列
- ``--limit``：每个系列最多处理 N 篇（调试用）
- ``--refresh``：覆盖同 title 的既有记录（默认幂等跳过）
- ``--demo``：建库后对该查询词跑一次检索 demo
- 许可证注意：研报版权归券商，本脚本产出仅供个人学习/研究
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pymupdf
except ImportError:
    print("错误：需要安装 PyMuPDF：pip install pymupdf")
    sys.exit(1)

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from quantmind.knowledge.store import KnowledgeStore  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SOURCE = "wiki-report"
REPORTS_DIR = PROJECT / "data_cache" / "reports"


def parse_report_index(md_path: Path) -> List[Dict]:
    """解析研报目录 md，提取标题和 PDF URL。

    返回：[{"title": "研报标题", "url": "https://...pdf", "series": "系列名"}, ...]
    """
    text = md_path.read_text(encoding="utf-8", errors="replace")
    reports = []
    
    # 匹配 markdown 链接：[标题](URL)
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)]+\.pdf)\)")
    for m in pattern.finditer(text):
        title = m.group(1).strip()
        url = m.group(2)
        # 从路径推断系列名
        series = md_path.parent.name
        reports.append({"title": title, "url": url, "series": series})
    
    return reports


def download_reports(reports: List[Dict], output_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """批量下载 PDF 到本地。

    返回：已下载的 PDF 文件路径列表（跳过已存在的）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    
    for i, r in enumerate(reports):
        if limit and i >= limit:
            break
        
        # 文件名：系列名_序号_标题.pdf（清理非法字符）
        safe_title = re.sub(r'[\\/*?:"<>|]', "_", r["title"])[:50]
        filename = f"{r['series']}_{i+1:02d}_{safe_title}.pdf"
        pdf_path = output_dir / filename
        
        if pdf_path.exists():
            print(f"  跳过已存在：{filename}")
            downloaded.append(pdf_path)
            continue
        
        print(f"  下载 [{i+1}/{len(reports)}]：{r['title'][:40]}...")
        try:
            urllib.request.urlretrieve(r["url"], pdf_path)
            downloaded.append(pdf_path)
            print(f"    ✓ {pdf_path.name}")
        except Exception as e:
            print(f"    ✗ 下载失败：{e}")
    
    return downloaded


def parse_pdf(pdf_path: Path) -> Dict:
    """用 PyMuPDF 解析 PDF，抽取全文文本。

    返回：{"text": "全文文本", "pages": N, "size_kb": N}
    """
    doc = pymupdf.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    
    full_text = "\n".join(text_parts)
    return {
        "text": full_text,
        "pages": len(text_parts),
        "size_kb": pdf_path.stat().st_size // 1024,
    }


def extract_structured_info(text: str, title: str) -> Dict:
    """启发式抽取结构化信息：因子名称、构造方法、IC/IR、样本期、适用市场。

    返回：{"factors": [...], "ic_ir": "...", "sample_period": "...", "market": "..."}
    """
    info = {
        "factors": [],
        "ic_ir": "",
        "sample_period": "",
        "market": "",
    }
    
    # 因子名称：标题中可能包含（如"特质波动率因子"、"市值因子"）
    factor_keywords = ["因子", "动量", "波动率", "市值", "价值", "质量", "流动性", "反转"]
    for kw in factor_keywords:
        if kw in title:
            info["factors"].append(kw)
    
    # IC/IR：查找 "IC"、"IR"、"信息系数" 等
    ic_pattern = re.compile(r"(IC|IR|信息系数)[^\d]*([\d.]+)", re.IGNORECASE)
    ic_matches = ic_pattern.findall(text[:2000])  # 只搜前 2000 字符
    if ic_matches:
        info["ic_ir"] = ", ".join(f"{m[0]}={m[1]}" for m in ic_matches[:3])
    
    # 样本期：查找 "20XX-20XX"、"20XX年-20XX年" 等
    period_pattern = re.compile(r"(20\d{2})[年/-]+(20\d{2})")
    period_matches = period_pattern.findall(text[:3000])
    if period_matches:
        info["sample_period"] = f"{period_matches[0][0]}-{period_matches[0][1]}"
    
    # 适用市场：查找 "A股"、"沪深"、"期货" 等
    market_keywords = ["A股", "沪深", "期货", "商品", "股指", "港股"]
    for kw in market_keywords:
        if kw in text[:2000]:
            info["market"] = kw
            break
    
    return info


def ingest_report(report: Dict, pdf_info: Dict, structured: Dict, store: KnowledgeStore) -> str:
    """入库为 methodology 记录。

    返回：kb_id
    """
    # 构造 content：全文前 5000 字符 + 结构化信息
    content_preview = pdf_info["text"][:5000]
    content = f"""# {report['title']}

## 结构化信息
- 因子：{', '.join(structured['factors']) or '未识别'}
- IC/IR：{structured['ic_ir'] or '未识别'}
- 样本期：{structured['sample_period'] or '未识别'}
- 适用市场：{structured['market'] or '未识别'}

## 研报摘要（前 5000 字符）
{content_preview}
"""
    
    # 构造 summary：标题 + 结构化信息一句话
    summary_parts = [report["title"]]
    if structured["factors"]:
        summary_parts.append(f"研究因子：{', '.join(structured['factors'])}")
    if structured["ic_ir"]:
        summary_parts.append(f"实证指标：{structured['ic_ir']}")
    summary = "。".join(summary_parts)
    
    # 构造 concept：一句话核心
    concept = f"{report['series']}系列研报：{report['title']}"
    
    # tags
    tags = ["研报", "因子", report["series"]]
    if structured["factors"]:
        tags.extend(structured["factors"])
    if structured["market"]:
        tags.append(structured["market"])
    
    # meta
    meta = {
        "kind": "report",
        "source_type": "quant-wiki-report",
        "license": "券商研报（仅供学习）",
        "pdf_url": report["url"],
        "pages": pdf_info["pages"],
        "size_kb": pdf_info["size_kb"],
        "factors": structured["factors"],
        "ic_ir": structured["ic_ir"],
        "sample_period": structured["sample_period"],
        "market": structured["market"],
    }
    
    kb_id = store.ingest_methodology(
        title=report["title"],
        concept=concept,
        summary=summary,
        content=content,
        source=SOURCE,
        tags=tags,
        meta=meta,
    )
    return kb_id


def _existing_titles(db_path: Path) -> set:
    """查 methodology 表中已存在的 (title) 集合（source='wiki-report'），用于幂等。"""
    if not db_path.exists():
        return set()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT title FROM methodology WHERE source=?", (SOURCE,)
        ).fetchall()
    return {r[0] for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="quant-wiki 研报 PDF → QuantMind 知识库（P1）")
    ap.add_argument("--wiki-dir", type=Path, required=True,
                    help="quant-wiki 仓库根目录（含 docs/advanced/研报精选）")
    ap.add_argument("--db", type=Path, default=None,
                    help="knowledge.db 路径（默认 quantmind/db/knowledge.db）")
    ap.add_argument("--series", type=str, default=None,
                    help="研报系列名称（如'中信-多因子系列'），不传则处理全部系列")
    ap.add_argument("--limit", type=int, default=None,
                    help="每个系列最多处理 N 篇（调试用）")
    ap.add_argument("--refresh", action="store_true",
                    help="覆盖同 title 的既有记录（默认幂等跳过）")
    ap.add_argument("--demo", type=str, default="",
                    help="建库后对该查询词跑一次检索 demo（空串跳过）")
    args = ap.parse_args()
    
    db_path = args.db or (PROJECT / "db" / "knowledge.db")
    wiki_dir = args.wiki_dir.resolve()
    reports_dir = wiki_dir / "docs" / "advanced" / "研报精选"
    
    if not reports_dir.is_dir():
        print(f"错误：未找到研报精选目录：{reports_dir}")
        return 1
    
    # 1. 解析研报目录
    print("=" * 60)
    print("步骤 1：解析研报目录")
    print("=" * 60)
    
    all_reports = []
    if args.series:
        # 只处理指定系列
        series_dir = reports_dir / args.series
        if not series_dir.is_dir():
            print(f"错误：未找到系列目录：{series_dir}")
            return 1
        index_md = series_dir / "index.md"
        if not index_md.exists():
            print(f"错误：未找到 index.md：{index_md}")
            return 1
        reports = parse_report_index(index_md)
        print(f"  {args.series}：{len(reports)} 篇研报")
        all_reports.extend(reports)
    else:
        # 处理全部系列
        for series_dir in sorted(reports_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            index_md = series_dir / "index.md"
            if not index_md.exists():
                continue
            reports = parse_report_index(index_md)
            print(f"  {series_dir.name}：{len(reports)} 篇研报")
            all_reports.extend(reports)
    
    if not all_reports:
        print("没有解析到任何研报")
        return 1
    
    print(f"\n总计：{len(all_reports)} 篇研报")
    
    # 2. 下载 PDF
    print("\n" + "=" * 60)
    print("步骤 2：下载 PDF")
    print("=" * 60)
    
    downloaded = download_reports(all_reports, REPORTS_DIR, limit=args.limit)
    print(f"\n已下载/已存在：{len(downloaded)} 篇")
    
    # 3. 解析 PDF 并入库
    print("\n" + "=" * 60)
    print("步骤 3：解析 PDF 并入库")
    print("=" * 60)
    
    store = KnowledgeStore(str(db_path))
    existing = _existing_titles(db_path)
    added = skipped = failed = 0
    
    for i, (report, pdf_path) in enumerate(zip(all_reports[:len(downloaded)], downloaded)):
        if report["title"] in existing and not args.refresh:
            print(f"  跳过已存在：{report['title'][:40]}")
            skipped += 1
            continue
        
        print(f"\n  [{i+1}/{len(downloaded)}] 解析：{report['title'][:40]}...")
        try:
            pdf_info = parse_pdf(pdf_path)
            structured = extract_structured_info(pdf_info["text"], report["title"])
            kb_id = ingest_report(report, pdf_info, structured, store)
            print(f"    ✓ 入库成功：kb_id={kb_id}")
            print(f"      因子：{structured['factors'] or '未识别'}")
            print(f"      IC/IR：{structured['ic_ir'] or '未识别'}")
            print(f"      样本期：{structured['sample_period'] or '未识别'}")
            print(f"      市场：{structured['market'] or '未识别'}")
            added += 1
        except Exception as e:
            print(f"    ✗ 解析失败：{e}")
            failed += 1
    
    print(f"\n总计：新增 {added}，幂等跳过 {skipped}，失败 {failed}")
    print(f"知识库：{db_path}")
    
    # 4. 检索 demo
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
