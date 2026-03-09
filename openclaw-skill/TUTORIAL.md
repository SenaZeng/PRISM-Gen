# PRISM_GEN_DEMO 使用教程

## 📚 目录
1. [快速开始](#快速开始)
2. [数据准备](#数据准备)
3. [基础查询](#基础查询)
4. [高级分析](#高级分析)
5. [可视化](#可视化)
6. [论文图表生成](#论文图表生成)
7. [批量处理](#批量处理)
8. [故障排除](#故障排除)

---

## 1. 快速开始

### 1.1 安装
```bash
# 克隆或下载项目
git clone <repository-url>
cd prism-gen-demo

# 运行安装脚本
bash setup.sh
```

### 1.2 基本测试
```bash
# 测试环境
python3 -c "import pandas; print('Pandas版本:', pandas.__version__)"

# 测试脚本
bash scripts/test_visualization.sh
```

### 1.3 添加数据
```bash
# 创建数据目录（如果不存在）
mkdir -p data

# 复制PRISM结果CSV文件
cp /path/to/your/prism/results/*.csv data/
```

---

## 2. 数据准备

### 2.1 支持的CSV格式
PRISM_GEN_DEMO支持标准的PRISM输出CSV格式，关键列包括：

| 列名 | 描述 | 类型 | 重要性 |
|------|------|------|--------|
| `smiles` | 分子SMILES表示 | 字符串 | 必需 |
| `pIC50` | 活性预测值 | 数值 | 重要 |
| `QED` | 药物相似性 | 数值 | 重要 |
| `LogP` | 脂水分配系数 | 数值 | 重要 |
| `MW` | 分子量 | 数值 | 重要 |
| `hERG_Prob` | hERG抑制风险 | 数值 | 重要 |
| `SA` | 合成可及性 | 数值 | 可选 |
| `TPSA` | 拓扑极性表面积 | 数值 | 可选 |

### 2.2 数据质量检查
```bash
# 查看数据源信息
bash scripts/demo_source_info.sh step4a_admet_final.csv

# 检查数据质量
bash scripts/test_full_functionality.sh
```

### 2.3 数据预处理建议
1. **统一列名**: 确保所有CSV文件的列名一致
2. **处理缺失值**: 脚本会自动处理，但建议预处理
3. **格式验证**: 确保数值列为正确类型

---

## 3. 基础查询

### 3.1 查看数据源
```bash
# 列出所有可用数据源
bash scripts/demo_list_sources.sh

# 查看单个数据源详细信息
bash scripts/demo_source_info.sh step4a_admet_final.csv
```

### 3.2 数据预览
```bash
# 预览前10行数据
bash scripts/demo_preview.sh step4a_admet_final.csv 10

# 简化版预览（不依赖pandas）
bash scripts/demo_simple_preview.sh step4a_admet_final.csv 5
```

### 3.3 条件过滤
```bash
# 简单条件过滤
bash scripts/demo_filter.sh step4a_admet_final.csv pIC50 '>' 7.0
bash scripts/demo_filter.sh step4a_admet_final.csv LogP '1.5-3.5' ''
bash scripts/demo_filter.sh step4a_admet_final.csv smiles '~' 'CC(=O)'

# 多条件过滤示例
# 先筛选高活性
bash scripts/demo_filter.sh step4a_admet_final.csv pIC50 '>' 7.0
# 再从结果中筛选低风险
bash scripts/demo_filter.sh filtered_result.csv hERG_Prob '<' 0.1
```

### 3.4 Top N筛选
```bash
# 按pIC50排序取Top 10
bash scripts/demo_top.sh step4a_admet_final.csv pIC50 10

# 按QED排序取Top 5（升序）
bash scripts/demo_top.sh step4a_admet_final.csv QED 5 asc

# 从过滤结果中取Top
bash scripts/demo_filter.sh step4a_admet_final.csv pIC50 '>' 7.0
bash scripts/demo_top.sh filtered_result.csv QED 10
```

---

## 4. 高级分析

### 4.1 分布分析
```bash
# 基本分布图
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50

# 带统计信息的分布图
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50 --stats --kde

# 自定义参数
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv LogP \
  --bins 20 \
  --title "LogP Distribution" \
  --output "logp_distribution.png" \
  --dpi 300
```

### 4.2 相关性分析
```bash
# 基本散点图
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv pIC50 QED

# 带趋势线和相关性
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv pIC50 QED \
  --trendline --correlation

# 分组分析
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv LogP pIC50 \
  --hue Active_Set \
  --title "LogP vs pIC50 by Activity"

# 多维度分析
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv MW LogP \
  --hue Active_Set \
  --size QED \
  --title "Multi-dimensional Analysis"
```

### 4.3 统计检验
所有可视化脚本自动包含：
- Pearson相关系数和p值
- Spearman秩相关系数
- 线性回归分析
- 描述性统计
- 异常值检测

---

## 5. 可视化

### 5.1 图表类型
| 图表类型 | 脚本 | 用途 |
|----------|------|------|
| 分布图 | `demo_plot_distribution.sh` | 单变量分布分析 |
| 散点图 | `demo_plot_scatter.sh` | 双变量相关性分析 |
| 组合图 | 自定义 | 多图表组合 |

### 5.2 输出格式
```bash
# PNG格式（默认）
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50 --format png

# PDF格式（矢量图）
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50 --format pdf

# SVG格式（矢量图）
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50 --format svg
```

### 5.3 质量控制
```bash
# 高分辨率输出（论文质量）
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50 \
  --dpi 300 \
  --output "figure1_high_res.png"

# 添加网格和统计信息
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv pIC50 QED \
  --grid --correlation --trendline
```

---

## 6. 论文图表生成

### 6.1 标准图表模板

#### 图1: 活性分布
```bash
bash scripts/demo_plot_distribution.sh step4a_admet_final.csv pIC50 \
  --title "Distribution of Predicted Activity (pIC50)" \
  --output "figure1_activity_distribution.png" \
  --dpi 300 --format png --stats --kde
```

#### 图2: 活性-性质相关性
```bash
bash scripts/demo_plot_scatter.sh step4a_admet_final.csv pIC50 QED \
  --title "Correlation between Activity and Drug-likeness" \
  --output "figure2_activity_qed_correlation.png" \
  --dpi 300 --format png --trendline --correlation --grid
```

#### 图3: 多指标分析
```bash
# 创建组合图的脚本示例
python3 -c "
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/step4a_admet_final.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 子图1: pIC50分布
axes[0,0].hist(df['pIC50'], bins=20, edgecolor='black')
axes[0,0].set_xlabel('pIC50')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('A) Activity Distribution')

# 子图2: QED分布
axes[0,1].hist(df['QED'], bins=20, edgecolor='black', color='orange')
axes[0,1].set_xlabel('QED')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('B) Drug-likeness Distribution')

# 子图3: pIC50 vs QED
axes[1,0].scatter(df['pIC50'], df['QED'], alpha=0.6)
axes[1,0].set_xlabel('pIC50')
axes[1,0].set_ylabel('QED')
axes[1,0].set_title('C) Activity vs Drug-likeness')

# 子图4: LogP分布
axes[1,1].boxplot(df['LogP'].dropna())
axes[1,1].set_ylabel('LogP')
axes[1,1].set_title('D) Lipophilicity Distribution')

plt.tight_layout()
plt.savefig('figure3_multi_analysis.png', dpi=300, bbox_inches='tight')
print('组合图已保存: figure3_multi_analysis.png')
"
```

### 6.2 统计表格生成
```bash
# 生成描述性统计表
python3 -c "
import pandas as pd

df = pd.read_csv('data/step4a_admet_final.csv')

# 选择关键列
key_columns = ['pIC50', 'QED', 'LogP', 'MW', 'hERG_Prob', 'SA', 'TPSA']
available_cols = [col for col in key_columns if col in df.columns]

if available_cols:
    stats = df[available_cols].describe().round(3)
    
    # 保存为CSV
    stats.to_csv('table1_descriptive_statistics.csv')
    print('统计表已保存: table1_descriptive_statistics.csv')
    
    # 打印LaTeX格式
    print('\\nLaTeX表格格式:')
    print(stats.to_latex())
else:
    print('无关键列数据')
"
```

### 6.3 图表说明模板
```
**Figure 1. Distribution of predicted activity values.**
(A) Histogram showing the distribution of pIC50 values for all compounds (n=200). 
(B) Box plot illustrating the range and quartiles of activity values. 
The vertical dashed line indicates the threshold for high activity (pIC50 > 7.0).

**Figure 2. Correlation between activity and drug-likeness.**
Scatter plot showing the relationship between pIC50 and QED scores (Pearson r = -0.598, p < 0.001). 
The red line represents the linear regression fit (R² = 0.358). 
Compounds in the upper right quadrant represent the most promising candidates.

**Table 1. Descriptive statistics of key molecular properties.**
Summary statistics for the 200 compounds analyzed, including mean, standard deviation, 
and range for each property. Values are presented as mean ± SD.
```

---

## 7. 批量处理

### 7.1 批量筛选
```bash
#!/bin/bash
# batch_filter.sh - 批量筛选脚本

# 定义筛选条件
CONDITIONS=(
  "pIC50 '>' 7.0"
  "QED '>' 0.6"
  "LogP '1.5-3.5' ''"
  "hERG_Prob '<' 0.1"
)

# 对每个条件执行筛选
for condition in "${CONDITIONS[@]}"; do
  echo "执行筛选: $condition"
  bash scripts/demo_filter.sh step4a_admet_final.csv $condition
done

echo "批量筛选完成"
```

### 7.2 批量可视化
```bash
#!/bin/bash
# batch_visualization.sh - 批量可视化脚本

# 定义要分析的列
COLUMNS=("pIC50" "QED" "LogP" "MW" "hERG_Prob")

# 为每列生成分布图
for column in "${COLUMNS[@]}"; do
  echo "生成 $column 分布图..."
  bash scripts/demo_plot_distribution.sh step4a_admet_final.csv "$column" \
    --output "${column}_distribution.png" \
    --dpi 150
done

# 生成相关性矩阵
echo "生成相关性矩阵..."
python3 -c "
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/step4a_admet_final.csv')
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
key_cols = [col for col in ['pIC50', 'QED', 'LogP', 'MW', 'hERG_Prob'] if col in numeric_cols]

if len(key_cols) > 1:
    corr_matrix = df[key_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Molecular Properties')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    print('相关性矩阵已保存: correlation_matrix.png')
"
```

### 7.3 结果汇总
```bash
#!/bin/bash
# generate_report.sh - 生成分析报告

echo "# PRISM结果分析报告" > report.md
echo "## 生成时间: $(date)" >> report.md
echo "" >> report.md

# 数据概览
echo "## 1. 数据概览" >> report.md
bash scripts/demo_list_sources.sh | grep -A5 "可用数据源" >> report.md
echo "" >> report.md

# 关键统计
echo "## 2. 关键统计" >> report.md
python3 -c "
import pandas as pd
df = pd.read_csv('data/step4a_admet_final.csv')

print('### 描述性统计')
print('| 属性 | 平均值 | 标准差 | 最小值 | 最大值 |')
print('|------|--------|--------|--------|--------|')

for col in ['pIC50', 'QED', 'LogP', 'MW', 'hERG_Prob']:
    if col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f'| {col} | {mean:.3f} | {std:.3f} | {min_val:.3f} | {max_val:.3f} |')
" >> report.md

echo "" >> report.md
echo "报告已生成: report.md"
```

---

## 8. 故障排除

### 8.1 常见问题

#### 问题1: Python包未安装
```
错误: ModuleNotFoundError: No module named 'pandas'
```
**解决方案**:
```bash
# 安装缺少的包
pip3 install pandas numpy matplotlib seaborn --user

# 或运行环境检查
bash scripts/_ensure_env.py
```

#### 问题2: 脚本无执行权限
```
错误: bash: scripts/demo_list_sources.sh: Permission denied
```
**解决方案**:
```bash
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

#### 问题3: 数据文件不存在
```
错误: 文件不存在: data/step4a_admet_final.csv
```
**解决方案**:
```bash
# 检查数据目录
ls -la data/

# 添加CSV文件
cp /path/to/your/csv/files/*.csv data/
```

#### 问题4: 列名不匹配
```
错误: 列不存在: pIC50
```
**解决方案**:
```bash
# 查看可用列
bash scripts/demo_source_info.sh your_file.csv

# 或直接查看文件
head -1 data/your_file.csv
```

### 8.2 性能优化

#### 大文件处理
```bash
# 使用简化版预览
bash scripts/demo_simple_preview.sh large_file.csv 10

# 分批处理
python3 -c "
import pandas as pd

# 分批读取大文件
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # 处理每个批次
    filtered = chunk[chunk['pIC50'] > 7.0]
    # 保存或进一步处理
"
```

#### 内存优化
```bash
# 指定数据类型减少内存使用
python3 -c "
import pandas as pd

dtypes = {
    'pIC50': 'float32',
    'QED': 'float32',
    'LogP': 'float32',
    'MW': 'float32',
    'hERG_Prob': 'float32'
}

df = pd.read_csv('data/step4a_admet_final.csv', dtype=dtypes)
print(f'内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
"
```

### 8.3 获取