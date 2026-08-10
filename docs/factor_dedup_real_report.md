# 真实数据多标的因子去冗余报告

> 数据：**20** 个真实期货主力连续，共同交易日 **300** 个。
> 因子：成功构建截面面板 **87** 个；Spearman 相关矩阵 + 贪心聚类（阈值 **0.7**，metric=|截面IC|），保留代表 **54** · 去冗余 **33**。

> **说明**：代表因子为互相低相关（<阈值）的独立 alpha 源，适合作组合构建输入；
  截面相关仍受期货主连池共性与样本期局限，非投资建议。

## 未纳入（失败/跳过）

- `alpha035(无截面样本)`
- `qlib_turnover_ratio(无截面样本)`
- `acad_liquidity_20(无截面样本)`

## 保留的代表因子（按 |截面IC| 降序）

| # | 代表因子 | 截面IC | |IC| | 聚类内高度相关成员 |
|---|---|---|---|---|
| 1 | `alpha041` | 0.0371 | 0.0371 | — |
| 2 | `acad_beta` | -0.0302 | 0.0302 | — |
| 3 | `acad_bab` | 0.0302 | 0.0302 | — |
| 4 | `open_interest_change_20` | -0.0273 | 0.0273 | — |
| 5 | `gtja191_020` | 0.0251 | 0.0251 | `qlib_bias_10`, `qlib_kdj_j`, `qlib_cci_20`, `gtja191_014` … (+3) |
| 6 | `alpha095` | -0.0244 | 0.0244 | — |
| 7 | `gtja191_038` | 0.0244 | 0.0244 | — |
| 8 | `mean_reversion_60` | -0.0231 | 0.0231 | `acad_short_term_reversal` |
| 9 | `gtja191_002` | -0.0211 | 0.0211 | — |
| 10 | `alpha026` | 0.0203 | 0.0203 | `gtja191_016`, `alpha050` |
| 11 | `alpha191_012` | -0.0202 | 0.0202 | — |
| 12 | `alpha054` | -0.0197 | 0.0197 | `alpha038` |
| 13 | `alpha028` | 0.0197 | 0.0197 | `alpha033` |
| 14 | `alpha002` | -0.0196 | 0.0196 | — |
| 15 | `alpha017` | -0.0192 | 0.0192 | — |
| 16 | `gtja191_013` | -0.0175 | 0.0175 | — |
| 17 | `alpha009` | 0.0172 | 0.0172 | — |
| 18 | `qlib_atr_14` | 0.0170 | 0.0170 | `qlib_boll_up`, `qlib_boll_low`, `qlib_ma_20`, `qlib_boll_mid` |
| 19 | `alpha023` | 0.0163 | 0.0163 | — |
| 20 | `momentum_60` | 0.0160 | 0.0160 | — |
| 21 | `alpha027` | -0.0159 | 0.0159 | — |
| 22 | `gtja191_012` | -0.0152 | 0.0152 | — |
| 23 | `qlib_mom_10` | -0.0151 | 0.0151 | `alpha191_007`, `qlib_macd_hist`, `gtja191_027`, `qlib_roc_12` … (+2) |
| 24 | `gtja191_026` | -0.0150 | 0.0150 | `alpha021` |
| 25 | `alpha101` | 0.0149 | 0.0149 | — |
| 26 | `acad_value_proxy` | 0.0141 | 0.0141 | — |
| 27 | `alpha012` | -0.0139 | 0.0139 | — |
| 28 | `acad_downside_vol` | -0.0137 | 0.0137 | `acad_vol_20`, `qlib_volatility_20`, `acad_idio_vol`, `volatility_20` |
| 29 | `qlib_rsi_14` | 0.0134 | 0.0134 | `qlib_kdj_d`, `momentum_20`, `qlib_macd_dif`, `gtja191_088` |
| 30 | `volume_change_5` | 0.0130 | 0.0130 | — |
| 31 | `gtja191_060` | -0.0124 | 0.0124 | — |
| 32 | `acad_mom_12m_1m` | 0.0101 | 0.0101 | — |
| 33 | `gtja191_006` | -0.0090 | 0.0090 | — |
| 34 | `gtja191_007` | -0.0089 | 0.0089 | — |
| 35 | `alpha075` | -0.0085 | 0.0085 | — |
| 36 | `term_structure_20` | -0.0084 | 0.0084 | — |
| 37 | `alpha006` | -0.0082 | 0.0082 | — |
| 38 | `gtja191_015` | -0.0082 | 0.0082 | — |
| 39 | `acad_skew_20` | 0.0076 | 0.0076 | — |
| 40 | `alpha191_081` | -0.0072 | 0.0072 | — |
| 41 | `alpha001` | 0.0071 | 0.0071 | — |
| 42 | `alpha093` | 0.0053 | 0.0053 | `gtja191_054` |
| 43 | `gtja191_037` | 0.0051 | 0.0051 | — |
| 44 | `gtja191_102` | 0.0047 | 0.0047 | — |
| 45 | `alpha042` | 0.0047 | 0.0047 | — |
| 46 | `qlib_macd_dea` | 0.0046 | 0.0046 | `acad_profit_growth` |
| 47 | `qlib_obv` | -0.0041 | 0.0041 | — |
| 48 | `qlib_wr_14` | -0.0038 | 0.0038 | — |
| 49 | `gtja191_096` | 0.0026 | 0.0026 | — |
| 50 | `alpha191_042` | 0.0018 | 0.0018 | — |
| 51 | `gtja191_033` | 0.0017 | 0.0017 | — |
| 52 | `gtja191_001` | -0.0008 | 0.0008 | — |
| 53 | `gtja191_045` | 0.0006 | 0.0006 | — |
| 54 | `alpha191_056` | -0.0005 | 0.0005 | — |

## 全部聚类（含被并入成员）

| 簇代表 | 簇大小 | |截面IC| | 成员 |
|---|---|---|---|
| `alpha041` | 1 | 0.0371 | `alpha041` |
| `acad_beta` | 1 | 0.0302 | `acad_beta` |
| `acad_bab` | 1 | 0.0302 | `acad_bab` |
| `open_interest_change_20` | 1 | 0.0273 | `open_interest_change_20` |
| `gtja191_020` | 8 | 0.0251 | `gtja191_020`, `qlib_bias_10`, `qlib_kdj_j`, `qlib_cci_20`, `gtja191_014`, `qlib_kdj_k`, `gtja191_003`, `gtja191_019` |
| `alpha095` | 1 | 0.0244 | `alpha095` |
| `gtja191_038` | 1 | 0.0244 | `gtja191_038` |
| `mean_reversion_60` | 2 | 0.0231 | `mean_reversion_60`, `acad_short_term_reversal` |
| `gtja191_002` | 1 | 0.0211 | `gtja191_002` |
| `alpha026` | 3 | 0.0203 | `alpha026`, `gtja191_016`, `alpha050` |
| `alpha191_012` | 1 | 0.0202 | `alpha191_012` |
| `alpha054` | 2 | 0.0197 | `alpha054`, `alpha038` |
| `alpha028` | 2 | 0.0197 | `alpha028`, `alpha033` |
| `alpha002` | 1 | 0.0196 | `alpha002` |
| `alpha017` | 1 | 0.0192 | `alpha017` |
| `gtja191_013` | 1 | 0.0175 | `gtja191_013` |
| `alpha009` | 1 | 0.0172 | `alpha009` |
| `qlib_atr_14` | 5 | 0.0170 | `qlib_atr_14`, `qlib_boll_up`, `qlib_boll_low`, `qlib_ma_20`, `qlib_boll_mid` |
| `alpha023` | 1 | 0.0163 | `alpha023` |
| `momentum_60` | 1 | 0.0160 | `momentum_60` |
| `alpha027` | 1 | 0.0159 | `alpha027` |
| `gtja191_012` | 1 | 0.0152 | `gtja191_012` |
| `qlib_mom_10` | 7 | 0.0151 | `qlib_mom_10`, `alpha191_007`, `qlib_macd_hist`, `gtja191_027`, `qlib_roc_12`, `gtja191_021`, `gtja191_079` |
| `gtja191_026` | 2 | 0.0150 | `gtja191_026`, `alpha021` |
| `alpha101` | 1 | 0.0149 | `alpha101` |
| `acad_value_proxy` | 1 | 0.0141 | `acad_value_proxy` |
| `alpha012` | 1 | 0.0139 | `alpha012` |
| `acad_downside_vol` | 5 | 0.0137 | `acad_downside_vol`, `acad_vol_20`, `qlib_volatility_20`, `acad_idio_vol`, `volatility_20` |
| `qlib_rsi_14` | 5 | 0.0134 | `qlib_rsi_14`, `qlib_kdj_d`, `momentum_20`, `qlib_macd_dif`, `gtja191_088` |
| `volume_change_5` | 1 | 0.0130 | `volume_change_5` |
| `gtja191_060` | 1 | 0.0124 | `gtja191_060` |
| `acad_mom_12m_1m` | 1 | 0.0101 | `acad_mom_12m_1m` |
| `gtja191_006` | 1 | 0.0090 | `gtja191_006` |
| `gtja191_007` | 1 | 0.0089 | `gtja191_007` |
| `alpha075` | 1 | 0.0085 | `alpha075` |
| `term_structure_20` | 1 | 0.0084 | `term_structure_20` |
| `alpha006` | 1 | 0.0082 | `alpha006` |
| `gtja191_015` | 1 | 0.0082 | `gtja191_015` |
| `acad_skew_20` | 1 | 0.0076 | `acad_skew_20` |
| `alpha191_081` | 1 | 0.0072 | `alpha191_081` |
| `alpha001` | 1 | 0.0071 | `alpha001` |
| `alpha093` | 2 | 0.0053 | `alpha093`, `gtja191_054` |
| `gtja191_037` | 1 | 0.0051 | `gtja191_037` |
| `gtja191_102` | 1 | 0.0047 | `gtja191_102` |
| `alpha042` | 1 | 0.0047 | `alpha042` |
| `qlib_macd_dea` | 2 | 0.0046 | `qlib_macd_dea`, `acad_profit_growth` |
| `qlib_obv` | 1 | 0.0041 | `qlib_obv` |
| `qlib_wr_14` | 1 | 0.0038 | `qlib_wr_14` |
| `gtja191_096` | 1 | 0.0026 | `gtja191_096` |
| `alpha191_042` | 1 | 0.0018 | `alpha191_042` |
| `gtja191_033` | 1 | 0.0017 | `gtja191_033` |
| `gtja191_001` | 1 | 0.0008 | `gtja191_001` |
| `gtja191_045` | 1 | 0.0006 | `gtja191_045` |
| `alpha191_056` | 1 | 0.0005 | `alpha191_056` |

## 高相关对（|corr| ≥ 0.7，跨簇冗余热点）

| |corr| | 因子A | 因子B | corr |
|---|---|---|---|
| 1.000 | `volatility_20` | `acad_vol_20` | 1.000 |
| 1.000 | `qlib_boll_mid` | `qlib_ma_20` | 1.000 |
| 1.000 | `momentum_20` | `gtja191_088` | 1.000 |
| -1.000 | `acad_bab` | `acad_beta` | -1.000 |
| 0.999 | `volatility_20` | `qlib_volatility_20` | 0.999 |
| 0.999 | `qlib_volatility_20` | `acad_vol_20` | 0.999 |
| 0.997 | `qlib_boll_low` | `qlib_ma_20` | 0.997 |
| 0.997 | `qlib_boll_low` | `qlib_boll_mid` | 0.997 |
| 0.997 | `qlib_boll_up` | `qlib_ma_20` | 0.997 |
| 0.997 | `qlib_boll_mid` | `qlib_boll_up` | 0.997 |
| 0.993 | `qlib_boll_low` | `qlib_boll_up` | 0.993 |
| -0.993 | `alpha033` | `alpha191_081` | -0.993 |
| 0.987 | `volatility_20` | `acad_idio_vol` | 0.987 |
| 0.987 | `acad_idio_vol` | `acad_vol_20` | 0.987 |
| 0.987 | `qlib_volatility_20` | `acad_idio_vol` | 0.987 |
| 0.951 | `alpha191_007` | `gtja191_021` | 0.951 |
| -0.944 | `momentum_20` | `acad_short_term_reversal` | -0.944 |
| -0.944 | `gtja191_088` | `acad_short_term_reversal` | -0.944 |
| 0.934 | `gtja191_096` | `qlib_kdj_k` | 0.934 |
| 0.919 | `alpha038` | `alpha054` | 0.919 |
| 0.907 | `gtja191_020` | `qlib_bias_10` | 0.907 |
| 0.906 | `gtja191_079` | `qlib_roc_12` | 0.906 |
| 0.904 | `gtja191_027` | `qlib_roc_12` | 0.904 |
| 0.902 | `qlib_macd_dea` | `qlib_macd_dif` | 0.902 |
| 0.897 | `qlib_volatility_20` | `acad_downside_vol` | 0.897 |
| 0.893 | `volatility_20` | `acad_downside_vol` | 0.893 |
| 0.893 | `acad_downside_vol` | `acad_vol_20` | 0.893 |
| 0.892 | `acad_downside_vol` | `acad_idio_vol` | 0.892 |
| -0.888 | `alpha191_081` | `gtja191_054` | -0.888 |
| 0.888 | `alpha033` | `gtja191_054` | 0.888 |
| 0.887 | `gtja191_096` | `qlib_kdj_d` | 0.887 |
| 0.886 | `qlib_kdj_d` | `qlib_kdj_k` | 0.886 |
| -0.885 | `mean_reversion_60` | `qlib_rsi_14` | -0.885 |
| 0.883 | `gtja191_019` | `qlib_bias_10` | 0.883 |
| -0.875 | `gtja191_014` | `gtja191_026` | -0.875 |
| 0.874 | `alpha050` | `gtja191_016` | 0.874 |
| -0.865 | `alpha095` | `acad_value_proxy` | -0.865 |
| 0.863 | `gtja191_019` | `gtja191_020` | 0.863 |
| 0.861 | `qlib_cci_20` | `qlib_rsi_14` | 0.861 |
| -0.859 | `qlib_kdj_k` | `qlib_wr_14` | -0.859 |
| 0.859 | `gtja191_027` | `qlib_kdj_d` | 0.859 |
| 0.854 | `qlib_atr_14` | `qlib_boll_up` | 0.854 |
| -0.847 | `alpha023` | `alpha075` | -0.847 |
| -0.846 | `alpha093` | `alpha191_081` | -0.846 |
| 0.844 | `alpha033` | `alpha093` | 0.844 |
| 0.842 | `qlib_atr_14` | `qlib_ma_20` | 0.842 |
| 0.842 | `qlib_atr_14` | `qlib_boll_mid` | 0.842 |
| 0.841 | `gtja191_079` | `qlib_kdj_d` | 0.841 |
| 0.838 | `gtja191_027` | `gtja191_079` | 0.838 |
| 0.836 | `qlib_bias_10` | `qlib_kdj_j` | 0.836 |
| -0.833 | `qlib_cci_20` | `qlib_wr_14` | -0.833 |
| 0.833 | `qlib_kdj_j` | `qlib_kdj_k` | 0.833 |
| 0.830 | `gtja191_079` | `qlib_rsi_14` | 0.830 |
| 0.827 | `qlib_atr_14` | `qlib_boll_low` | 0.827 |
| 0.826 | `gtja191_014` | `gtja191_019` | 0.826 |
| 0.823 | `qlib_cci_20` | `qlib_kdj_k` | 0.823 |
| 0.821 | `gtja191_027` | `gtja191_096` | 0.821 |
| 0.821 | `gtja191_027` | `qlib_kdj_k` | 0.821 |
| 0.819 | `alpha026` | `gtja191_016` | 0.819 |
| 0.815 | `alpha191_007` | `qlib_mom_10` | 0.815 |
| 0.810 | `gtja191_079` | `qlib_cci_20` | 0.810 |
| -0.809 | `qlib_bias_10` | `qlib_wr_14` | -0.809 |
| 0.808 | `gtja191_079` | `qlib_kdj_k` | 0.808 |
| 0.808 | `gtja191_021` | `qlib_mom_10` | 0.808 |
| -0.808 | `alpha028` | `gtja191_013` | -0.808 |
| -0.807 | `qlib_kdj_j` | `qlib_wr_14` | -0.807 |
| -0.800 | `alpha075` | `alpha101` | -0.800 |
| 0.798 | `gtja191_020` | `qlib_kdj_j` | 0.798 |
| 0.797 | `momentum_20` | `qlib_rsi_14` | 0.797 |
| 0.797 | `gtja191_088` | `qlib_rsi_14` | 0.797 |
| -0.796 | `gtja191_079` | `qlib_wr_14` | -0.796 |
| 0.794 | `qlib_roc_12` | `qlib_rsi_14` | 0.794 |
| 0.791 | `alpha191_007` | `qlib_macd_hist` | 0.791 |
| -0.790 | `qlib_rsi_14` | `qlib_wr_14` | -0.790 |
| 0.788 | `qlib_cci_20` | `qlib_roc_12` | 0.788 |
| 0.785 | `alpha093` | `gtja191_054` | 0.785 |
| 0.785 | `gtja191_027` | `qlib_cci_20` | 0.785 |
| -0.785 | `qlib_rsi_14` | `acad_short_term_reversal` | -0.785 |
| 0.785 | `qlib_kdj_d` | `qlib_roc_12` | 0.785 |
| 0.782 | `gtja191_027` | `qlib_rsi_14` | 0.782 |
| 0.778 | `gtja191_019` | `qlib_kdj_j` | 0.778 |
| -0.777 | `mean_reversion_60` | `qlib_macd_dif` | -0.777 |
| 0.777 | `qlib_macd_hist` | `qlib_mom_10` | 0.777 |
| 0.775 | `qlib_kdj_k` | `qlib_rsi_14` | 0.775 |
| 0.774 | `qlib_bias_10` | `qlib_cci_20` | 0.774 |
| 0.773 | `qlib_bias_10` | `qlib_kdj_k` | 0.773 |
| 0.770 | `gtja191_021` | `qlib_macd_hist` | 0.770 |
| 0.767 | `gtja191_021` | `gtja191_096` | 0.767 |
| -0.767 | `gtja191_026` | `qlib_bias_10` | -0.767 |
| -0.763 | `qlib_roc_12` | `qlib_wr_14` | -0.763 |
| 0.763 | `qlib_kdj_k` | `qlib_roc_12` | 0.763 |
| 0.761 | `mean_reversion_60` | `acad_short_term_reversal` | 0.761 |
| -0.760 | `gtja191_003` | `gtja191_026` | -0.760 |
| 0.759 | `alpha191_007` | `gtja191_096` | 0.759 |
| 0.758 | `qlib_kdj_d` | `qlib_rsi_14` | 0.758 |
| 0.758 | `alpha191_007` | `qlib_kdj_k` | 0.758 |
| 0.757 | `qlib_cci_20` | `qlib_kdj_d` | 0.757 |
| 0.756 | `gtja191_003` | `gtja191_014` | 0.756 |
| -0.756 | `momentum_20` | `mean_reversion_60` | -0.756 |
| -0.756 | `mean_reversion_60` | `gtja191_088` | -0.756 |
| -0.755 | `gtja191_019` | `gtja191_026` | -0.755 |
| 0.754 | `gtja191_079` | `gtja191_096` | 0.754 |
| 0.752 | `gtja191_096` | `qlib_kdj_j` | 0.752 |
| 0.747 | `gtja191_021` | `gtja191_027` | 0.747 |
| -0.745 | `alpha038` | `alpha093` | -0.745 |
| 0.739 | `gtja191_014` | `qlib_bias_10` | 0.739 |
| 0.739 | `gtja191_096` | `qlib_cci_20` | 0.739 |
| 0.738 | `gtja191_027` | `qlib_mom_10` | 0.738 |
| -0.737 | `gtja191_096` | `qlib_wr_14` | -0.737 |
| 0.736 | `gtja191_020` | `qlib_kdj_k` | 0.736 |
| 0.731 | `alpha191_007` | `gtja191_027` | 0.731 |
| 0.731 | `gtja191_021` | `qlib_kdj_k` | 0.731 |
| 0.731 | `alpha038` | `alpha191_081` | 0.731 |
| -0.730 | `qlib_kdj_d` | `qlib_wr_14` | -0.730 |
| -0.730 | `gtja191_020` | `qlib_wr_14` | -0.730 |
| -0.729 | `alpha033` | `alpha038` | -0.729 |
| 0.727 | `gtja191_003` | `gtja191_020` | 0.727 |
| 0.724 | `qlib_bias_10` | `qlib_rsi_14` | 0.724 |
| -0.723 | `gtja191_027` | `qlib_wr_14` | -0.723 |
| 0.720 | `qlib_kdj_k` | `qlib_mom_10` | 0.720 |
| 0.720 | `gtja191_014` | `gtja191_020` | 0.720 |
| 0.719 | `gtja191_020` | `qlib_cci_20` | 0.719 |
| -0.717 | `mean_reversion_60` | `qlib_cci_20` | -0.717 |
| 0.717 | `qlib_mom_10` | `qlib_roc_12` | 0.717 |
| 0.715 | `alpha026` | `alpha050` | 0.715 |
| 0.715 | `gtja191_079` | `qlib_mom_10` | 0.715 |
| -0.714 | `gtja191_020` | `gtja191_026` | -0.714 |
| 0.714 | `qlib_cci_20` | `qlib_mom_10` | 0.714 |
| 0.714 | `gtja191_096` | `qlib_roc_12` | 0.714 |
| 0.714 | `qlib_macd_dea` | `acad_profit_growth` | 0.714 |
| 0.711 | `gtja191_021` | `qlib_kdj_d` | 0.711 |
| -0.710 | `gtja191_026` | `qlib_kdj_j` | -0.710 |
| 0.709 | `qlib_cci_20` | `qlib_kdj_j` | 0.709 |
| 0.709 | `alpha028` | `alpha033` | 0.709 |
| 0.709 | `gtja191_014` | `qlib_kdj_j` | 0.709 |
| -0.707 | `gtja191_019` | `qlib_wr_14` | -0.707 |
| 0.706 | `qlib_macd_dif` | `qlib_rsi_14` | 0.706 |
| -0.705 | `alpha028` | `alpha191_081` | -0.705 |
| -0.704 | `qlib_macd_dif` | `acad_short_term_reversal` | -0.704 |
| 0.704 | `alpha021` | `gtja191_026` | 0.704 |

## 生成信息

- 脚本：`scripts/factor_dedup_real.py`
- 标的：rb0, IF0, cu0, m0, ta0, i0, au0, y0, j0, al0, zn0, SR0, CF0, MA0, L0, FU0, a0, bu0, pp0, eg0
- 共同交易日：300 · 相关阈值：0.7 · metric：|截面IC|
- 生成时间：2026-08-10 10:04:46