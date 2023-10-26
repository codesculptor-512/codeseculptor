# 提示学习（prompt learning）

## 什么是提示学习

### 提示学习的背景

- 预训练模型中已经存在知识

- 预训练模型本身具有少样本的学习能力（incontext learning）

![](<新建 Markdown_md_files/180394e0-73ca-11ee-906c-5777a7c2d163.jpeg?v=1&type=image>)

### 提示学习的本质

- 将所有的下游任务统一成预训练任务

- 以特定的模版，将下游任务的数据组装成自然语言形式，充分挖掘预训练模型本身的能力

  ![](<新建 Markdown_md_files/9faab360-73c5-11ee-906c-5777a7c2d163.jpeg?v=1&type=image>)

- 提示模版的作用在于：将训练数据组装成自然语言的形式，并在合适的位置 MASK，以激发预训练模型的能力

- Prompt Tuning：通过构建提示学习样本，只需要少量数据的 Prompt Tuning，就可以达到很好的效果，具有较强的零样本/少样本学习能力

### 提示学习的组成部分

- **提示模版（添加提示）**：根据使用预训练模型，构建完形填空 or 基于前缀生成两种类别的模版

  特效非常炫酷，我很喜欢。这是一部 _\[MASK]_ 电影

  特效非常炫酷，我很喜欢。这部电影很 _\[MASK]_&#x20;

- **预训练语言模型**（答案搜索）

- **类别映射/Verbalizer:** 根据经验选择合适的类别映射词

- ![](<新建 Markdown_md_files/3d186de0-73da-11ee-ae65-f3f10b416556.jpeg?v=1&type=image>)

## 为什么需要提示学习？

- PLM+Fine-tuning 形式存在缺陷

  模型在做预训练时采用的任务形式是自回归、自编码。与下游的任务形式存在明显的区别。（下游 CLS）需要较多的数据来适应新的场景。导致过拟合/学习能力差

- 模型越来越大，为了一个特定的模型去 finetuning 然后部署，会导致资源的极大浪费（无法通用到不同的任务中）

## 预训练的语言模型

**标准语言模型（SLM）的目标正是这样做的，训练模型以优化训练语料库中文本的概率**$P(\boldsymbol{x})$\*\* 。在这些情况下，文本通常以\*\*自回归的方式进行预测，一次一个地预测序列中的标记。这通常是从左到右进行的（如下所述），但也可以按其他顺序进行。

标准 LM 目标的一个流行替代方案是去噪目标，该目标将一些去噪函数 $\tilde{\boldsymbol{x}}=f_{\text {noise }}(\boldsymbol{x})$ 应用于输入句子，然后在给出带噪文本$P(\boldsymbol{x} \mid \tilde{\boldsymbol{x}})$的情况下，尝试预测原始输入句子。这些目标有两种常见的风格：

\*\*受损文本重建（CTR）\*\*这些目标通过仅计算输入句子中有噪声部分的损失，将处理后的文本恢复到其未受损状态。

\*\*全文重构（FTR）\*\*这些目标通过计算整个输入文本的损失来重构文本，无论输入文本是否有噪声。

预训练的 LMs 的主要训练目标在确定其对特定提示任务的适用性方面起着重要作用。例如，从左到右的自回归 LMs 可能特别适合**前缀提示（prefix prompts）**，而重建目标可能更适合完形**填空提示（cloze prompts）**。此外，使用标准 LM 和 FTR 目标训练的模型可能更适合文本生成任务，而其他任务（如分类）可以使用使用任何这些目标训练的模型来制定。

## 提示设计（prompt engineering）

### 提示类型（prompt shape）

提示有两种主要类型：*cloze prompt*用于填充文本字符串的空白；prefix prompt，用于延续字符串前缀。选择哪一个取决于任务和用于解决任务的模型。通常，对于与生成相关的任务，或使用标准自回归 LM 解决的任务，前缀提示往往更有用，因为它们与模型的从左到右的特性很好地匹配。对于使用掩码 LMs 解决的任务，完形填空提示非常适合，因为它们与训练前任务的形式非常匹配。全文重构模型更通用，可以与完形填空或前缀提示一起使用。最后，对于一些涉及多个输入的任务，如文本对分类，提示模板必须包含两个输入的空间，\[X1]和\[X2]，或更多。

### 手动模板设计（Manual Template Engineering）

创建提示（prompt）最自然的方法是基于人类的内省手动创建直观的模板。例如，开创性的 LAMA 数据集提供了手动创建的完形填空模板，以探索 LMs 中的知识。Brown 等人（2020 年）创建手工制作的前缀提示（prompt），以处理各种任务，包括问答、翻译和常识推理的探测任务。Schick 和 Sch̉utze（2020、2021a、b）在文本分类和有条件文本生成任务的 few-shot learning 设置中使用预定义模板。

### 自动模板设计（Automated Template Learning）

#### 离散提示（Discrete Prompts）

- 提示挖掘（Prompt Mining）

- 提示改写（Prompt Paraphrasing）

- 提示生成（Prompt Generation）

- 提示打分（Prompt Scoring）

- 基于梯度的搜索（Gradient-based Search）

#### 连续提示（Continuous Prompts）

- 前缀调整（Prefix Tuning）

- 硬软混合调整（Hard-SoftPrompt Hybrid Tuning）

-

## 应答设计（Prompt Answer Engineering）

与为提示方法设计适当输入的提示设计不同，应答设计旨在搜索答案空间$\mathcal{Z}$和原始输出$\mathcal{Y}$的映射，从而生成有效的预测模型。图 1 的“应答设计”部分说明了在执行答案工程时必须考虑的两个方面：确定答案粒度和选择答案设计方法。

- 手动设计（Manual Design）答案空间

  - 无限空间（Unconstrained Spaces）

  - 有限空间（Constrained Spaces）

- 离散答案搜索（Discrete Answer Search）

- 连续答案搜索（Continuous Answer Search）

## 多提示（\*\*Multi-Prompt）\*\*学习

- 提示集成

- 提示扩增

- 提示合成

- 提示分解

![](<新建 Markdown_md_files/5d50df70-73da-11ee-ae65-f3f10b416556.jpeg?v=1&type=image>)

## 应用

**文本分类（Text Classification）**  对于文本分类任务，以前的大多数工作都使用完形填空提示，提示设计（Gao 等人，2021；Hambardzumyan 等人，2021；Lester 等人，2021）和答案设计（Schick 和 Sch̉utze，2021a；Schick 等人，2020；Gao 等人，2021）都得到了广泛的探索。大多数现有研究探索了在“固定提示 LM 调整”策略（定义见§7.2.4）的 few-show 设置下，提示学习对文本分类的有效性。

\*\*自然语言推理（Natural Language Inference-NLI）\*\*NLI 旨在预测两个给定句子的关系（例如，蕴涵（entailment））。与文本分类任务类似，对于自然语言推理任务，通常使用完形填空提示（Schick 和 Sch̉utze，2021a）。关于提示设计，研究人员主要关注在 few-shot 学习环境中的模板搜索，答案空间  �  通常是从词汇表中手动预先选择的。

。。。。。。

###
