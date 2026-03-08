# PRISM-Gen

## English

**PRISM-Gen** is a physics-informed multi-stage computational framework for the discovery of **broad-spectrum coronavirus main protease (Mpro) inhibitors**.

The framework integrates generative molecular design, surrogate modeling, multi-fidelity electronic-structure screening, and multi-target docking to prioritize candidate molecules with predicted cross-strain binding competence.

This repository provides the implementation of the **PRISM-Gen pipeline** used for computational prioritization of candidate inhibitors targeting coronavirus Mpro orthologs.

---

## 中文

**PRISM-Gen** 是一个物理信息驱动（physics-informed）的多阶段计算框架，用于发现 **广谱冠状病毒主蛋白酶（Mpro）抑制剂**。

该框架整合了生成式分子设计、代理模型预测、多保真电子结构筛选以及多靶点分子对接等方法，用于优先筛选具有潜在跨病毒株结合能力的候选分子。

本仓库提供 **PRISM-Gen pipeline** 的实现，用于对冠状病毒 Mpro 靶点的候选抑制剂进行计算优先级筛选。

---

# Overview / 项目概述

## English

The PRISM-Gen workflow follows a hierarchical multi-stage screening strategy designed to balance exploration of chemical space with physical and structural validation.

The pipeline consists of the following stages:

1. **Fragment-based molecular generation**  
   Candidate molecules are generated using a fragment-tree variational autoencoder (FRATTVAE).

2. **Surrogate-guided activity prediction**  
   A machine-learning surrogate model based on Uni-Mol representations estimates predicted inhibitory activity.

3. **Semi-empirical electronic screening (GFN2-xTB)**  
   Frontier orbital energies and electrostatic descriptors are computed to identify electronically unstable molecules.

4. **Density Functional Theory (DFT) validation**  
   Selected candidates are evaluated using DFT calculations (e.g., B3LYP/6-31G*) to refine electronic descriptors.

5. **ADMET filtering**  
   Drug-likeness and pharmacokinetic properties are evaluated.

6. **Broad-spectrum multi-target docking**  
   Molecules are docked against Mpro structures from multiple coronaviruses  
   (SARS-CoV-2, SARS-CoV-1, and MERS-CoV).

7. **Molecular dynamics validation**  
   Top candidates are further validated using molecular dynamics simulations.

---

## 中文

PRISM-Gen 工作流程采用 **分层多阶段筛选策略**，在化学空间探索与物理结构验证之间取得平衡。

该 pipeline 包含以下主要步骤：

1. **基于片段的分子生成**  
   使用 fragment-tree 变分自编码器（FRATTVAE）生成候选分子。

2. **代理模型活性预测**  
   基于 Uni-Mol 表征的机器学习模型预测候选分子的潜在抑制活性。

3. **半经验电子结构筛选（GFN2-xTB）**  
   计算前线轨道能级和电荷分布等电子结构描述符，以识别电子结构异常的分子。

4. **密度泛函理论（DFT）验证**  
   对筛选后的候选分子进行 DFT 计算（如 B3LYP/6-31G*），以获得更精确的电子结构信息。

5. **ADMET 性质筛选**  
   评估候选分子的药物相似性和药代动力学性质。

6. **多靶点广谱分子对接**  
   将候选分子分别对接到多个冠状病毒的 Mpro 蛋白结构  
   （SARS-CoV-2、SARS-CoV-1、MERS-CoV）。

7. **分子动力学验证**  
   对排名靠前的候选分子进行分子动力学模拟验证其结合稳定性。

---

# Key Features / 主要特点

## English

- Physics-informed generative molecular design  
- Multi-fidelity electronic-structure screening  
- Gaussian Electronic Moderation (GEM) scoring  
- Distributionally robust multi-target docking  
- Broad-spectrum antiviral prioritization  

## 中文

- 物理信息驱动的生成式分子设计  
- 多保真电子结构筛选  
- Gaussian Electronic Moderation (GEM) 评分机制  
- 分布鲁棒（distributionally robust）的多靶点对接策略  
- 广谱抗病毒候选分子优先筛选  

---

# Repository Structure / 仓库结构


## License

This repository is released for **academic research purposes only**.

Any commercial use of this software requires explicit permission from the author.

## 许可协议

本仓库仅用于 **学术研究用途**。

任何商业用途均需获得作者明确许可。