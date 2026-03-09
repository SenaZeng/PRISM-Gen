# PRISM_GEN_DEMO - 药物发现预计算结果展示技能

![PRISM Demo](https://img.shields.io/badge/PRISM-GEN_DEMO-blue)
![OpenClaw Skill](https://img.shields.io/badge/OpenClaw-Skill-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📋 概述

PRISM_GEN_DEMO是一个专为药物发现研究设计的OpenClaw技能，用于展示和分析PRISM-Gen预计算结果。本技能提供对分子生成/筛选的多个CSV结果文件进行检索、过滤、排序、合并和可视化的完整功能，特别适合论文发表和研究成果展示。

## 🎯 设计理念

### 核心目标
- **可移植性**：不触发HPC计算流程，只处理既有CSV文件
- **稳定性**：避免网络依赖和计算不确定性
- **查询型**：提供检索、过滤、排序、合并功能
- **结构化**：以清晰的结构化方式返回结果
- **可视化**：提供数据可视化和profile汇总

### 论文发表标准
- ✅ 数据真实性：仅使用真实PRISM计算结果
- ✅ 方法透明：所有处理步骤可追溯
- ✅ 结果可重现：提供完整代码和数据
- ✅ 统计合理：使用标准统计方法
- ✅ 可视化清晰：提供多种图表类型

## 📁 项目结构

```
prism-gen-demo/
├── README.md                    # 本文档
├── SKILL.md                     # OpenClaw技能定义
├── requirements.txt             # Python依赖
├── data/                        # 预计算结果CSV文件
│   ├── step3a_optimized_molecules.csv
│   ├── step4a_admet_final.csv
│   └── ...
├── scripts/                     # 核心脚本
│   ├── demo_list_sources.sh     # 列出数据源
│   ├── demo_source_info.sh      # 数据源信息
│   ├── demo_preview.sh          # 数据预览
│   ├── demo_filter.sh           # 数据过滤
│   ├── demo_top.sh              # Top N筛选
│   ├── demo_plot_distribution.sh # 分布图
│   ├── demo_plot_scatter.sh     # 散点图
│   ├── _python_wrapper.sh       # Python环境包装
│   └── _ensure_env.py           # 环境检查
├── config/                      # 配置文件
├── examples/                    # 使用示例
├── docs/                        # 文档
├── output/                      # 输出目录
│   ├── filtered/                # 过滤结果
│   └── top/                     # Top N结果
└── plots/                       # 图表输出
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆或下载本技能
git clone <repository-url>
cd prism-gen-demo

# 安装Python依赖
pip install -r requirements.txt
# 或使用conda
conda create -n prism-demo python=3.10 pandas numpy matplotlib seaborn
conda activate prism-demo
```

### 2. 准备数据

将PRISM预计算结果CSV文件放入`data/`目录：

```bash
# 示例文件命名
cp /path/to/results/*.csv data/
```

### 3. 基本使用

```bash
# 查看可用数据源
bash scripts/demo_list_sources.sh

# 查看数据源详细信息
bash scripts/demo_source_info.sh step4a_admet_final.csv

# 预览数据
bash scripts/demo_preview.sh step4a_admet_final.csv 10

# 筛选高活性分子
bash scripts/demo_filter.sh step4a_admet_final.csv pIC50 '>' 7.0

# 获取Top 10活性分子
bash scripts/demo_top.sh step4a_admet_final.csv pIC50 10

# 生成分布图
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50

# 生成散点图分析相关性
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv pIC50 QED --trendline --correlation
```

## 📊 核心功能

### 1. 数据管理
- **数据源列表**：查看所有可用CSV文件
- **数据预览**：显示文件结构和样本数据
- **数据统计**：计算关键指标的统计信息

### 2. 数据查询
- **条件过滤**：基于单列或多列条件筛选分子
- **Top N筛选**：按指定列排序获取最佳分子
- **范围查询**：支持数值范围和字符串匹配

### 3. 数据分析
- **相关性分析**：计算Pearson、Spearman相关系数
- **回归分析**：线性回归和趋势线
- **分布分析**：直方图、箱线图、Q-Q图

### 4. 数据可视化
- **分布图**：单变量分布可视化
- **散点图**：双变量相关性分析
- **统计图表**：论文质量的统计图表

### 5. 数据导出
- **CSV导出**：保存过滤和排序结果
- **图表导出**：PNG、PDF、SVG格式
- **报告生成**：结构化分析报告

## 🔬 数据模型

### 支持的CSV文件
- `step3a_optimized_molecules.csv` - 代理模型优化分子
- `step3b_dft_results.csv` - xTB+DFT电子筛选结果
- `step3c_dft_refined.csv` - GEM重排序结果
- `step4a_admet_final.csv` - ADMET过滤结果
- `step4b_top_molecules_pyscf.csv` - DFT验证(PySCF)结果
- `step4c_master_summary.csv` - 主汇总表
- `step5a_broadspectrum_docking.csv` - 广谱对接结果
- `step5b_final_candidates.csv` - 最终候选分子

### 关键分子属性
- **标识符**：smiles, molecule_id, name
- **活性**：pIC50, reward, broad
- **物化性质**：LogP, MW, TPSA, HBD, HBA
- **安全性**：hERG_Prob, AMES, hepatotoxicity
- **药物相似性**：QED, SA, Lipinski
- **电子性质**：gap, energy, dipole
- **对接结果**：docking_score, binding_energy

## 📈 使用示例

### 示例1：高活性分子筛选

```bash
# 筛选pIC50 > 7.0的高活性分子
bash scripts/demo_filter.sh step4a_admet_final.csv pIC50 '>' 7.0

# 进一步筛选低hERG风险的分子
bash scripts/demo_filter.sh filtered_result.csv hERG_Prob '<' 0.1

# 按QED排序获取Top 10
bash scripts/demo_top.sh filtered_result.csv QED 10
```

### 示例2：多指标分析

```bash
# 分析pIC50分布
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50 --kde --stats

# 分析pIC50与QED的相关性
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv pIC50 QED --trendline --correlation

# 按活性分组分析
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv LogP pIC50 --hue Active_Set
```

### 示例3：论文图表生成

```bash
# 生成高质量的论文图表
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50 \
  --title "Distribution of pIC50 Values" \
  --output "figure1_pIC50_distribution.png" \
  --dpi 300 --format png

# 生成相关性分析图
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv pIC50 QED \
  --title "Correlation between Activity and Drug-likeness" \
  --trendline --correlation --grid \
  --output "figure2_correlation.png" \
  --dpi 300 --format png
```

## 📝 论文发表支持

### 统计方法
- **相关性检验**：Pearson、Spearman相关系数
- **显著性检验**：p值计算和解释
- **回归分析**：线性回归方程和R²
- **分布检验**：正态性检验（Shapiro-Wilk）

### 图表标准
- **分辨率**：支持300 DPI高质量输出
- **格式**：PNG、PDF、SVG多种格式
- **标注**：自动添加统计信息和图例
- **风格**：学术期刊标准图表风格

### 数据报告
- **统计摘要**：关键指标的描述性统计
- **质量检查**：缺失值、异常值分析
- **筛选记录**：完整的过滤和排序历史
- **可重复性**：所有步骤可追溯和重现

## 🔧 技术细节

### 环境要求
- **Python**: 3.10+
- **核心包**: pandas, numpy, matplotlib, seaborn
- **可选包**: scipy, scikit-learn, plotly

### 性能特点
- **内存优化**：流式处理大文件
- **缓存机制**：加速重复查询
- **并行处理**：支持多核处理（可选）
- **错误恢复**：完善的错误处理和日志

### 扩展性
- **插件系统**：支持自定义分析模块
- **配置驱动**：通过配置文件扩展功能
- **API接口**：提供Python API供其他工具调用

## 📚 文档

### 详细文档
- [安装指南](docs/installation.md)
- [使用教程](docs/tutorial.md)
- [API参考](docs/api.md)
- [故障排除](docs/troubleshooting.md)

### 示例文档
- [完整工作流示例](examples/full_workflow.md)
- [论文图表生成](examples/paper_figures.md)
- [批量处理示例](examples/batch_processing.md)

## 🤝 贡献指南

### 报告问题
请在GitHub Issues中报告bug或提出功能建议。

### 提交代码
1. Fork本仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

### 开发规范
- 遵循PEP 8代码风格
- 添加单元测试
- 更新相关文档
- 保持向后兼容性

## 📄 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- **PRISM-Gen团队**：提供药物发现计算框架
- **OpenClaw社区**：优秀的AI助手平台
- **所有贡献者**：感谢你们的代码和反馈

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [项目地址](https://github.com/yourusername/prism-gen-demo)
- 电子邮件: your.email@example.com
- 文档网站: [文档地址](https://yourusername.github.io/prism-gen-demo)

---

*最后更新: 2026年3月5日*