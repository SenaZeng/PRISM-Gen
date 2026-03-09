---
name: prism-gen-demo
title: "PRISM-Gen Demo: Drug Discovery Pre-result Analysis / 药物发现预计算结果展示"
description: |
  English: Retrieve, filter, sort, merge, and visualize multiple CSV result files from PRISM-Gen molecular generation/screening. Provides portable query-based skills. No HPC connection required, directly analyze pre-calculated results.
  中文: 对PRISM-Gen分子生成/筛选的多个CSV结果文件进行检索、过滤、排序、合并和可视化，提供可移植的查询型技能。无需HPC连接，直接分析预计算结果。
author: "May"
version: "1.0.2"
license: "MIT"
tags: ["drug-discovery", "chemistry", "data-analysis", "visualization", "csv", "molecular", "screening", "药物发现", "化学", "数据分析", "可视化"]
categories: ["science", "data-analysis", "chemistry", "科学", "数据分析", "化学"]
openclaw:
  emoji: "🧪"
  min_version: "0.10.0"
  dependencies: ["pandas", "numpy", "matplotlib", "seaborn", "scipy", "scikit-learn"]
  platforms: ["linux", "macos", "wsl"]
  skill_type: "analysis"
  requires_data: true
clawhub:
  published: true
  featured: false
  verified: false
  downloads: 0
  rating: 0
  last_updated: "2026-03-09"
  repository: ""
  issues: ""
  documentation: ""
---
# PRISM_GEN_DEMO Skill / PRISM_GEN_DEMO 技能

**English**: PRISM-Gen pre-calculation result display demo: Provides retrieval, filtering, sorting, merging, and visualization of multiple CSV result files from molecular generation/screening, offering portable query-based skills.

**中文**: PRISM-Gen预计算结果展示Demo：对分子生成/筛选的多个CSV结果文件进行检索、过滤、排序、合并和可视化，提供可移植的查询型技能。

## Design Goals / 设计目标

- **Portability / 可移植性**: Does not trigger HPC computation workflows, only processes existing CSV files / 不触发HPC计算流程，只处理既有CSV文件
- **Stability / 稳定性**: Core functions work offline; Python dependencies optional for advanced visualization / 核心功能离线工作；Python依赖仅用于高级可视化（可选）
- **Query-based / 查询型**: Provides retrieval, filtering, sorting, and merging functions / 提供检索、过滤、排序、合并功能
- **Structured / 结构化**: Returns results in a clear structured format / 以清晰的结构化方式返回结果
- **Visualization / 可视化**: Provides data visualization and profile summarization (requires Python) / 提供数据可视化和profile汇总（需要Python）

## Usage Scenarios / 使用场景

✅ **Use this skill when / 使用此技能当：**
- "Show PRISM Demo results" / "查看PRISM Demo结果" / "展示预计算结果"
- "Retrieve molecular data" / "Filter CSV results" / "检索分子数据" / "过滤CSV结果"
- "Sort molecules" / "Top N screening" / "排序分子" / "Top N筛选"
- "Merge multiple stage results" / "Profile summarization" / "合并多个阶段结果" / "Profile汇总"
- "Visualization analysis" / "Chart display" / "可视化分析" / "图表展示"
- "Export results" / "Format conversion" / "导出结果" / "格式转换"

## Core Function Architecture / 核心功能架构

### 1. Data Source Management / 数据源管理
- **List data sources / 列出数据源**: Display all available CSV files / 显示所有可用CSV文件
- **Source information / 数据源信息**: Show file structure and statistics / 显示文件结构和统计信息
- **Data preview / 数据预览**: Quick view of sample data / 快速查看样本数据

### 2. Data Query / 数据查询
- **Conditional filtering / 条件过滤**: Single or multi-column condition screening / 基于单列或多列条件筛选分子
- **Top N selection / Top N筛选**: Sort by specified column to get best molecules / 按指定列排序获取最佳分子
- **Range queries / 范围查询**: Support numerical ranges and string matching / 支持数值范围和字符串匹配

### 3. Data Analysis / 数据分析
- **Correlation analysis / 相关性分析**: Calculate Pearson, Spearman correlation coefficients / 计算Pearson、Spearman相关系数
- **Regression analysis / 回归分析**: Linear regression and trend lines / 线性回归和趋势线
- **Distribution analysis / 分布分析**: Histograms, box plots, Q-Q plots / 直方图、箱线图、Q-Q图

### 4. Data Visualization / 数据可视化
- **Distribution plots / 分布图**: Univariate distribution visualization / 单变量分布可视化
- **Scatter plots / 散点图**: Bivariate correlation analysis / 双变量相关性分析
- **Statistical charts / 统计图表**: Publication-quality statistical charts / 论文质量的统计图表

### 5. Data Export / 数据导出
- **CSV export / CSV导出**: Save filtered and sorted results / 保存过滤和排序结果
- **Chart export / 图表导出**: PNG, PDF, SVG formats / PNG、PDF、SVG格式
- **Report generation / 报告生成**: Structured analysis reports / 结构化分析报告

## Supported CSV Files / 支持的CSV文件

- `step3a_optimized_molecules.csv` - Surrogate model optimized molecules / 代理模型优化分子
- `step3b_dft_results.csv` - xTB+DFT electronic screening results / xTB+DFT电子筛选结果
- `step3c_dft_refined.csv` - GEM re-ranking results / GEM重排序结果
- `step4a_admet_final.csv` - ADMET filtering results / ADMET过滤结果
- `step4b_top_molecules_pyscf.csv` - DFT validation (PySCF) results / DFT验证(PySCF)结果
- `step4c_master_summary.csv` - Master summary table / 主汇总表
- `step5a_broadspectrum_docking.csv` - Broad-spectrum docking results / 广谱对接结果
- `step5b_final_candidates.csv` - Final candidate molecules / 最终候选分子

## Key Molecular Properties / 关键分子属性

- **Identifiers / 标识符**: smiles, molecule_id, name
- **Activity / 活性**: pIC50, reward, broad
- **Physicochemical properties / 物化性质**: LogP, MW, TPSA, HBD, HBA
- **Safety / 安全性**: hERG_Prob, AMES, hepatotoxicity
- **Drug-likeness / 药物相似性**: QED, SA, Lipinski
- **Electronic properties / 电子性质**: gap, energy, dipole
- **Docking results / 对接结果**: docking_score, binding_energy

## Quick Start Examples / 快速开始示例

### Example 1: List Data Sources / 示例1：列出数据源
```bash
bash scripts/demo_list_sources.sh
```

### Example 2: Filter High-Activity Molecules / 示例2：筛选高活性分子
```bash
# Filter molecules with pIC50 > 7.0 / 筛选pIC50 > 7.0的分子
bash scripts/demo_filter.sh step4a_admet_final.csv pIC50 '>' 7.0
```

### Example 3: Get Top 10 Active Molecules / 示例3：获取Top 10活性分子
```bash
bash scripts/demo_top.sh step4a_admet_final.csv pIC50 10
```

### Example 4: Generate Distribution Plot / 示例4：生成分布图
```bash
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50
```

### Example 5: Correlation Analysis / 示例5：相关性分析
```bash
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv pIC50 QED --trendline --correlation
```

## Technical Requirements / 技术要求

### Basic Functions (No Installation Required, Fully Offline) / 基础功能（无需安装，完全离线）
- ✅ Bash shell environment / Bash shell环境
- ✅ Standard Unix tools (awk, sed, grep) / 标准Unix工具
- ✅ File read/write permissions / 文件读写权限
- 🚫 **No network connection required** / **无需网络连接**
- 🚫 **No Python installation required** / **无需Python安装**

### Advanced Functions (Requires Python, Works Offline After Installation) / 高级功能（需要Python，安装后离线工作）
- 🐍 Python 3.10+ / Python 3.10+
- 📦 Core packages: pandas, numpy, matplotlib, seaborn / 核心包
- 🔬 Scientific computing: scipy, scikit-learn (optional) / 科学计算包（可选）
- ⚠️ **Network required for installation only** / **仅安装需要网络**
- ✅ **Offline usage after installation** / **安装后可离线使用**

### Compatibility / 兼容性
- ✅ Linux / macOS / WSL2
- ✅ Local file system / 本地文件系统
- ✅ Basic functions: Fully offline / 基础功能：完全离线
- ⚠️  Advanced functions: Network required for installation only / 高级功能：仅安装需要网络
- ✅ Linux / macOS / WSL2
- ✅ Local file system / 本地文件系统
- ✅ Basic functions: Fully offline / 基础功能：完全离线
- ⚠️ Advanced functions: Network required for installation only / 高级功能：仅安装需要网络

## Project Structure / 项目结构
```
prism-gen-demo/
├── README.md                    # This document / 本文档
├── SKILL.md                     # OpenClaw skill definition / OpenClaw技能定义
├── requirements.txt             # Python dependencies / Python依赖
├── data/                        # Pre-calculation result CSV files / 预计算结果CSV文件
├── scripts/                     # Core scripts / 核心脚本
├── config/                      # Configuration files / 配置文件
├── examples/                    # Usage examples / 使用示例
├── docs/                        # Documentation / 文档
├── output/                      # Output directory / 输出目录
└── plots/                       # Chart output / 图表输出
```

## License / 许可证
MIT License - See [LICENSE](LICENSE) file for details. / MIT许可证 - 详见[LICENSE](LICENSE)文件。

## Contact / 联系方式
For questions or suggestions, please refer to the documentation or contact the skill author. / 如有问题或建议，请参考文档或联系技能作者。