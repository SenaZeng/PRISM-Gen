# OpenClaw Skill: PRISM-Gen Pipeline

This directory provides the **OpenClaw skill configuration** for running the PRISM-Gen pipeline.

The skill is published on **ClawHub**, where users can directly access and execute the workflow in the OpenClaw environment.

## ClawHub Link

The PRISM-Gen OpenClaw skill is available at:

<YOUR-CLAW-SKILL-LINK>

## Overview

PRISM-Gen is a physics-informed multi-stage pipeline designed for the discovery of **broad-spectrum coronavirus Mpro inhibitors**.

The workflow integrates the following stages:

1. Fragment-based generative molecular design  
2. Surrogate-guided activity prediction  
3. Semi-empirical electronic screening (GFN2-xTB)  
4. Density Functional Theory (DFT) validation for selected candidates  
5. ADMET filtering and drug-likeness evaluation  
6. Broad-spectrum multi-target docking across coronavirus Mpro proteins  

The DFT stage refines electronic-structure descriptors for high-priority candidates before docking evaluation.

This OpenClaw skill allows users to run the PRISM-Gen workflow directly through the OpenClaw platform.

## Usage

1. Open the ClawHub link above.  
2. Import the skill into your OpenClaw workspace.  
3. Configure input parameters if needed.  
4. Execute the workflow to generate and evaluate candidate molecules.

For detailed methodological descriptions, please refer to the main repository and the associated manuscript.

---

# OpenClaw 技能：PRISM-Gen Pipeline

本目录提供 **PRISM-Gen 流水线的 OpenClaw 技能配置**。

该技能已发布在 **ClawHub** 平台上，用户可以在 OpenClaw 环境中直接导入并运行该工作流程。

## ClawHub 链接

PRISM-Gen 的 OpenClaw 技能可通过以下链接访问：

<YOUR-CLAW-SKILL-LINK>

## 项目简介

PRISM-Gen 是一个 **物理信息驱动（physics-informed）的多阶段分子筛选框架**，用于发现具有 **广谱抗冠状病毒 Mpro 活性的小分子抑制剂**。

该工作流程包含以下主要步骤：

1. 基于片段的生成式分子设计  
2. 代理模型驱动的活性预测  
3. 半经验电子结构筛选（GFN2-xTB）  
4. 密度泛函理论（DFT）计算验证  
5. ADMET 性质过滤与成药性评估  
6. 针对多个冠状病毒 Mpro 的广谱 docking 评估  

其中 **DFT 步骤用于在 docking 之前对候选分子的电子结构进行高精度验证和筛选**。

通过 OpenClaw 技能，用户可以在 OpenClaw 平台中直接运行 PRISM-Gen 工作流程。

## 使用方法

1. 打开 ClawHub 链接：https://clawhub.ai/SenaZeng/prism-gen-demo 
2. 将该技能导入 OpenClaw 工作空间  
3. 根据需要配置输入参数  
4. 运行工作流程以生成并评估候选分子  

更多算法细节和方法说明，请参考主仓库及相关论文。
