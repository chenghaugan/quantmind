"""测试 efinance 数据获取能力

验证：
1. 能否正常获取数据
2. 分钟数据实际能获取多少根
3. 与 akshare 对比数据量
"""
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def test_efinance():
    """测试 efinance 数据获取"""
    try:
        import efinance as ef
        print("[OK] efinance 导入成功")
    except ImportError:
        print("[FAIL] efinance 未安装，请先安装: pip install efinance")
        return
    
    # 股指期货行情 ID 映射
    # 格式：{symbol: quote_id}
    QUOTE_IDS = {
        "IF0": "8.IF0",  # 沪深300主力
        "IC0": "8.IC0",  # 中证500主力
        "IH0": "8.IH0",  # 上证50主力
        "IM0": "8.IM0",  # 中证1000主力
    }
    
    print("\n" + "=" * 70)
    print("测试 efinance 数据获取能力")
    print("=" * 70)
    
    results = {}
    
    for symbol, quote_id in QUOTE_IDS.items():
        print(f"\n【{symbol}】行情ID: {quote_id}")
        print("-" * 70)
        
        symbol_results = {}
        
        # 测试不同周期
        periods = [
            (1, "1分钟"),
            (5, "5分钟"),
            (15, "15分钟"),
            (30, "30分钟"),
            (60, "60分钟"),
            (101, "日线"),
        ]
        
        for klt, period_name in periods:
            try:
                df = ef.futures.get_quote_history(quote_id, klt=klt)
                
                if df is None or df.empty:
                    print(f"  {period_name:8s}: 无数据")
                    symbol_results[period_name] = {"count": 0, "start": None, "end": None}
                    continue
                
                count = len(df)
                
                # 获取时间范围
                date_col = None
                for col in df.columns:
                    if '日期' in col or 'date' in col.lower():
                        date_col = col
                        break
                
                if date_col and count > 0:
                    start = df.iloc[0][date_col]
                    end = df.iloc[-1][date_col]
                else:
                    start = end = "N/A"
                
                print(f"  {period_name:8s}: {count:6d} 根 | {start} ~ {end}")
                
                symbol_results[period_name] = {
                    "count": count,
                    "start": str(start),
                    "end": str(end),
                    "df": df  # 保存 DataFrame 用于后续分析
                }
                
            except Exception as e:
                print(f"  {period_name:8s}: 错误 - {str(e)[:60]}")
                symbol_results[period_name] = {"count": 0, "error": str(e)}
        
        results[symbol] = symbol_results
    
    # 对比总结
    print("\n" + "=" * 70)
    print("数据量对比总结")
    print("=" * 70)
    print(f"{'品种':<8s} | {'1分钟':<10s} | {'5分钟':<10s} | {'15分钟':<10s} | {'30分钟':<10s} | {'60分钟':<10s} | {'日线':<10s}")
    print("-" * 70)
    
    for symbol, symbol_results in results.items():
        row = f"{symbol:<8s}"
        for period_name in ["1分钟", "5分钟", "15分钟", "30分钟", "60分钟", "日线"]:
            count = symbol_results.get(period_name, {}).get("count", 0)
            row += f" | {count:<10d}"
        print(row)
    
    # 保存样本数据用于分析
    print("\n" + "=" * 70)
    print("保存样本数据")
    print("=" * 70)
    
    sample_dir = PROJECT / "data_cache" / "efinance_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol, symbol_results in results.items():
        for period_name, data in symbol_results.items():
            if "df" in data and data["count"] > 0:
                df = data["df"]
                filename = f"{symbol}_{period_name.replace('分钟', 'm').replace('日线', '1d')}.csv"
                df.to_csv(sample_dir / filename, index=False, encoding="utf-8-sig")
                print(f"  [OK] 已保存: {filename} ({data['count']} 根)")
    
    print(f"\n样本数据已保存到: {sample_dir}")
    
    return results


def compare_with_akshare():
    """对比 akshare 和 efinance 的数据量"""
    print("\n" + "=" * 70)
    print("对比 akshare vs efinance")
    print("=" * 70)
    
    try:
        import akshare as ak
        import efinance as ef
    except ImportError as e:
        print(f"[FAIL] 缺少依赖: {e}")
        return
    
    symbol = "IF0"
    print(f"\n测试品种: {symbol}")
    print("-" * 70)
    
    # akshare 数据量（从现有 parquet 文件读取）
    parquet_path = PROJECT / "data_cache" / f"{symbol}.CFFEX.1m.parquet"
    if parquet_path.exists():
        df_ak = pd.read_parquet(parquet_path)
        ak_count = len(df_ak)
        ak_start = df_ak.index[0] if len(df_ak) > 0 else None
        ak_end = df_ak.index[-1] if len(df_ak) > 0 else None
        print(f"akshare (1分钟): {ak_count} 根 | {ak_start} ~ {ak_end}")
    else:
        print(f"akshare (1分钟): 无本地数据")
        ak_count = 0
    
    # efinance 数据量
    try:
        df_ef = ef.futures.get_quote_history("8.IF0", klt=1)
        ef_count = len(df_ef) if df_ef is not None else 0
        if ef_count > 0:
            # 找到日期列
            date_col = None
            for col in df_ef.columns:
                if '日期' in col or 'date' in col.lower():
                    date_col = col
                    break
            
            if date_col:
                ef_start = df_ef.iloc[0][date_col]
                ef_end = df_ef.iloc[-1][date_col]
                print(f"efinance (1分钟): {ef_count} 根 | {ef_start} ~ {ef_end}")
            else:
                print(f"efinance (1分钟): {ef_count} 根 | 无法解析时间")
        else:
            print(f"efinance (1分钟): 无数据")
    except Exception as e:
        print(f"efinance (1分钟): 错误 - {str(e)[:60]}")
        ef_count = 0
    
    # 对比
    if ak_count > 0 and ef_count > 0:
        ratio = ef_count / ak_count
        print(f"\n对比结果: efinance 数据量是 akshare 的 {ratio:.2f} 倍")
        if ratio > 1.5:
            print("[OK] efinance 明显优于 akshare（分钟数据）")
        elif ratio > 1.0:
            print("[OK] efinance 略优于 akshare（分钟数据）")
        else:
            print("[WARN] akshare 数据量更大或相当")
    
    return {"akshare": ak_count, "efinance": ef_count}


if __name__ == "__main__":
    print("efinance 集成测试")
    print("=" * 70)
    
    # 测试 efinance 基本能力
    results = test_efinance()
    
    # 对比 akshare
    if results:
        compare_with_akshare()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
