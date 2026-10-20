# Recognition Tasks

# Model Evaluation: ImageNet-Pretrained ConvNeXt-Tiny(Pretrained)

## Experimental Setup

This experiment investigated whether an ImageNet-pretrained ConvNeXt-Tiny could be adapted to classify ADNI MRI images as Alzheimer's disease (AD) or cognitively normal (CN). The original classifier was replaced with a two-class classifier. Training used two stages: the classifier head was trained for two epochs while the feature extractor was frozen; the whole network was then fine-tuned for six epochs.

The dataset contained 30,520 image slices from 1,526 scans and 680 participants. It was divided at the participant level into 21,700 training images, 4,320 validation images, and 4,500 test images. No missing image files were found, and participants did not overlap between splits.

ConvNeXt is a modernised convolutional network, not a Transformer with self-attention. Its design was developed by updating standard ConvNet components while retaining a convolutional architecture. In this experiment, the feature extractor began with ImageNet-1K pretrained weights; only the final classification layer was changed for the AD/CN task.

## Test Results

The checkpoint with the lowest validation loss was selected before test evaluation. It was saved at the first epoch of full fine-tuning, with a validation loss of 0.3890. The held-out test results were:

| Evaluation level | Samples | Accuracy | Balanced accuracy | Macro-F1 | AUROC | AD precision | AD recall |
|-----------------|---------|----------|-------------------|----------|-------|--------------|-----------|
| Image | 4,500 | 0.8604 | 0.8623 | 0.8604 | 0.9386 | 0.8205 | 0.9079 |
| Scan | 225 | 0.9156 | 0.9174 | 0.9155 | 0.9645 | 0.8740 | 0.9630 |
| Patient | 102 | 0.8922 | 0.9124 | 0.8840 | 0.9574 | 0.7619 | 0.9697 |

The image-level test accuracy was 86.04%, exceeding the project's 80% target. Patient-level accuracy was 89.22%, also above that target. This supports the conclusion that the adapted pretrained ConvNeXt-Tiny performed well on this held-out split. However, slices from the same participant are correlated, so image-level accuracy alone would give an incomplete picture; patient-level metrics are also important.

At the patient level, the confusion matrix was:

| | Predicted CN | Predicted AD |
|----------------|-------------|--------------|
| **Actual CN** | 59 | 10 |
| **Actual AD** | 1 | 32 |

The model correctly identified 32 of 33 AD participants, but incorrectly classified 10 of 69 CN participants as AD. Thus, AD recall was high, while AD precision was lower. The result shows a trade-off between detecting AD cases and generating false alarms.

## Training Behaviour and Error Analysis

During full fine-tuning, training accuracy rose to 96.46% by epoch 6, while validation loss increased from 0.3890 at epoch 1 to 0.5487 at epoch 6. This divergence suggests overfitting. Selecting the checkpoint by validation loss helped avoid using the final, more overfit epoch.

The patient-level predictions identify false positives including 041_S_6192 (predicted AD probability 0.9288) and 941_S_6080 (0.9043), and a false negative, 135_S_4863 (0.3048). These cases are candidates for the required visual error review. Prediction files alone do not show why these errors occurred, so their MRI slices and scans must be inspected before attributing causes. The high probabilities for two false positives also mean that raw model confidence should not be assumed to be reliable. Any confidence threshold or referral rule should be selected using validation data, not tuned on the test set.

## Limitations and Recommendation

The experiment ran on an NVIDIA A100-SXM4-40GB GPU. The binary-classifier version of ConvNeXt-Tiny has approximately 27.82 million trainable parameters. The supplied logs do not record peak GPU memory, total training time, or inference latency; these operational measurements remain to be reported.

This experiment demonstrates that ImageNet-pretrained ConvNeXt-Tiny, after replacing its classifier and fine-tuning on ADNI MRI images, exceeded the test-accuracy target. It does not establish that ConvNeXt is better than a standard CNN baseline, because the baseline has not yet been evaluated. It also does not satisfy the tutor's restriction on pretrained models. The final compliant comparison must therefore use ConvNeXt-Tiny initialised with weights=None and the ordinary CNN baseline on the same participant-level split, then compare their test metrics and resource costs.

## 当前报告能支持的结论
预训练 ConvNeXt-Tiny 在这次测试集上达到了 86.04% 图像级准确率和 89.22% 患者级准确率，都高于 0.80

# Training Memos: ConvNeXt-Tiny vs CNN Baseline (From Scratch)

## Memo 1: First Formal Training (8 epochs)

### Experimental Setup
- Both **ConvNeXt-Tiny** and a **parameter-matched CNN baseline** were trained from random weight initialization (`weights=None`).
- Shared configuration: same participant-level data split, AdamW optimizer, learning rate 0.001, 8 epochs.
- Dataset: 30,520 image slices, 1,526 scans, 680 participants.
- Split: 21,700 training / 4,320 validation / 4,500 test images (participant-level, no overlap).
- Parameter count difference between the two models: **0.192%**.

### Image-Level Results

| Metric | ConvNeXt-Tiny | CNN Baseline |
|--------|--------------|--------------|
| Parameter count | 27,821,666 | 27,875,297 |
| Lowest validation loss | 0.5464 (epoch 2) | 0.6906 (epoch 7) |
| Highest validation accuracy | 0.7468 (epoch 6) | 0.7588 (epoch 7) |
| Training accuracy (epoch 8) | 0.8310 | 0.8970 |
| Validation accuracy (epoch 8) | 0.7306 | 0.7275 |
| Training time | 6.35 min | 6.26 min |
| Peak GPU memory allocated | 4.12 GB | 0.80 GB |

### Preliminary Analysis
- Both models' highest image-level validation accuracy was **below 0.80**.
- CNN's peak accuracy was 1.20 percentage points higher than ConvNeXt's, but this is a single run and insufficient to conclude the CNN architecture is superior overall.
- ConvNeXt achieved a lower minimum validation loss; its best checkpoint (selected by validation loss) came from **epoch 2**, not the highest-accuracy epoch 6.
- CNN's best checkpoint came from **epoch 7**.
- By epoch 8, training accuracy continued rising while validation accuracy fell below each model's peak, indicating **generalization did not improve** with further training.

---

## Memo 2: Second Training — Extended Epochs for ConvNeXt-Tiny

### Strategy
- ConvNeXt-Tiny initialized randomly; data split, AdamW, and learning rate (0.001) unchanged.
- Only change: training epochs increased from **8 to 30**.
- The first 8 epochs' logs are identical to Memo 1; lowest validation loss still at epoch 2 (0.5464).
- Best checkpoint by validation loss therefore unchanged.

### Results

| Metric | 8 epochs | 30 epochs |
|--------|----------|------------|
| Highest image-level validation accuracy | 0.7468 (epoch 6) | 0.7507 (epoch 14) |
| Training accuracy (final epoch) | 0.8310 | 0.9818 |
| Validation loss (final epoch) | 0.7112 | 1.4096 |
| Training time | 6.35 min | 23.19 min |

### Analysis
- Extending training improved highest image-level validation accuracy by only **0.39 percentage points**, still below 0.80.
- Meanwhile, training accuracy reached **98.18%** and validation loss rose sharply, indicating **clear overfitting**.
- Conclusion: simply increasing training epochs did not effectively improve generalization; insufficient training length was **not** the main limitation.
- Next step: revert to 8 epochs and separately test **warmup** and **cosine decay** scheduling.
- To conserve compute, the baseline was not re-run in this round.

---

## Reporting Caution

The results above are **training and image-level validation** outcomes only. They do **not** justify claims that the model's test-set or patient-level accuracy is below 0.80; such conclusions require formal evaluation of the corresponding checkpoints on the held-out test set.


## Training Memo 3: ConvNeXt-Tiny with Warmup + Cosine Decay (From Scratch)

## Experimental Setup

- Model: ConvNeXt-Tiny, initialized from random weights (`weights=None`)
- Data split: same participant-level split as previous runs
- Optimizer: AdamW, base learning rate 0.001, weight decay 0.0001
- Schedule: **2-step linear warmup** followed by **cosine decay** (new, relative to first 8-epoch run)
- Epochs: 8
- No new CNN baseline results in this round (ConvNeXt only)

## Image-Level Validation Metrics

| Metric | First 8-epoch (fixed LR) | 30-epoch (fixed LR) | This run (warmup + cosine) |
|--------|------------------------|---------------------|----------------------------|
| Lowest validation loss (epoch) | 0.5464 (epoch 2) | 0.5464 (epoch 2) | **0.5393 (epoch 3)** |
| Highest validation accuracy (epoch) | 0.7468 (epoch 6) | 0.7507 (epoch 14) | 0.7340 (epoch 7) |
| Best checkpoint saved (by val loss) | epoch 2 | epoch 2 | **epoch 3** |
| Validation accuracy at best checkpoint | 0.7315 | 0.7315 | 0.7238 |
| Training accuracy (final epoch) | 0.8310 | 0.9818 | 0.8839 |
| Training time | 6.35 min | 23.19 min | 6.70 min |

## Analysis

- Compared to the first 8-epoch run, lowest validation loss **decreased by 0.0071** (0.5464 → 0.5393), but highest validation accuracy **dropped by 1.28 percentage points** (0.7468 → 0.7340).
- The actual saved best checkpoint's validation accuracy also **decreased by 0.77 percentage points** (0.7315 → 0.7238).
- Training accuracy reached 0.8839 by epoch 8, while final-epoch validation loss was 0.7585 and validation accuracy was 0.7229 — the gap between training and validation performance persists, indicating this schedule did not resolve generalization issues.
- The 30-epoch fixed-LR experiment's best checkpoint also remained at epoch 2, supporting the earlier conclusion that **simply extending training does not solve the problem**.

## Reportable Conclusions

- In this single run, warmup + cosine decay **slightly reduced best image-level validation loss** but **did not improve validation accuracy** and **did not reach 0.80**.
- This result evaluates the **combined effect** of warmup and cosine decay together; it cannot separately attribute effectiveness (or lack thereof) to either component.
- Patient-level validation metrics are **not yet available**; no claim can be made about whether the patient-level target (≥ 0.80) is met based on these results alone.

- ## Training Memo 4: Fixed Learning Rate Comparison (0.001 vs 3e-4)

## Experimental Setup

- Model: ConvNeXt-Tiny, initialized from random weights (`weights=None`)
- Data split: same participant-level split (21,700 train / 4,320 val / 4,500 test)
- Optimizer: AdamW
- Epochs: 8
- Variables tested: **base learning rate only** (all other settings held constant)
- Previous optimizations (warmup, cosine decay, label smoothing) were **removed** to isolate the effect of learning rate alone
- No new CNN baseline results in this round

## Results

| Metric | LR = 0.001 (fixed) | LR = 3e-4 (fixed) | Change |
|--------|-------------------|-------------------|--------|
| Lowest validation loss (epoch) | 0.5464 (epoch 2) | 0.5497 (epoch 2) | +0.0033 |
| Highest validation accuracy (epoch) | 0.7315 (epoch 2) | 0.7310 (epoch 2) | -0.0005 |
| Training accuracy (epoch 2) | 0.6840 | — | — |
| Training accuracy (epoch 8) | 0.8429 | — | — |
| Validation accuracy (epoch 8) | 0.7074 | — | — |
| Training time | ~6.3 min | ~6.3 min | — |

## Analysis

### Learning rate reduction had negligible effect
- Dropping LR from 0.001 to 3e-4 changed best validation accuracy by **only 0.05 percentage points** (0.7315 → 0.7310).
- Best validation loss also showed minimal change (0.5464 → 0.5497).
- This indicates that **learning rate magnitude is not the primary bottleneck** in the current setup.

### Generalization gap persists
- At LR = 0.001, training accuracy rose from 0.6840 (epoch 2) to 0.8429 (epoch 8), while validation accuracy at epoch 8 dropped to 0.7074.
- The model continues to fit training data without corresponding improvement on validation — consistent with **overfitting / insufficient model regularization**, not a learning rate problem.
- The fact that the best checkpoint is saved at epoch 2 (by validation loss) does **not** prove the learning rate is too large; it indicates the model begins overfitting early regardless of LR choice within this range.

### Implication for optimization direction
- Multiple optimization strategies (warmup, cosine decay, label smoothing, LR reduction) have each produced **< 1 percentage point** change in best validation metrics.
- This pattern suggests the limitation is **architectural or data-level**, not an optimization hyperparameter issue.

## Next Steps (Proposed)

| Priority | Action | Rationale |
|----------|--------|-----------|
| 1 | **Visual inspection of processed slices and error cases** | Confirm cropping does not discard critical brain regions; verify train/val preprocessing is consistent. Zero compute cost. |
| 2 | **Increase weight decay** (0.0001 → 0.01), LR = 0.001, 8 epochs | ConvNeXt's original ImageNet config used strong weight decay (0.05–0.1). Even 0.01 is a reasonable test on MRI data. Targets overfitting directly. |
| 3 | **Multi-slice input** (adjacent slices as 3 channels) | More informative than duplicating grayscale; adds anatomical context. Larger data-processing change, deferred until after steps 1–2. |

## Reportable Conclusions

- Reducing learning rate from 0.001 to 3e-4 **did not meaningfully change** best validation loss or accuracy.
- The persistent generalization gap across all LR variants (fixed, warmup+cosine, reduced) indicates **overfitting is the dominant issue**, not step-size selection.
- Further hyperparameter tuning of learning rate alone is unlikely to reach the 0.80 target; **regularization (weight decay) and data representation** are more promising directions.
- All results remain at the **image-level validation** stage; no claim about test-set or patient-level performance can be made until the best checkpoint is formally evaluated.


## 2.5D Input Experiment Report

## 1. Experimental Results

### 中文版

为引入相邻切片的空间信息，我们将 ConvNeXt-Tiny 的输入从“同一张灰度 MRI 切片复制为三个通道”改为“前一张、当前张、后一张切片分别作为三个通道”。模型仍从零训练，并保持患者级数据划分、学习率 0.001、权重衰减 0.0001 和 8 轮训练不变。按最低验证损失选取的最佳模型来自第 3 轮，验证损失为 0.5513。

在默认分类阈值 0.5 下，2.5D 模型的图像、扫描和患者级验证准确率分别为 74.19%、78.24% 和 79.41%，高于原始单切片 ConvNeXt 的 73.15%、76.85% 和 74.51%。然而，患者级 AD recall 从 **78.79%（26/33）**降至 57.58%（19/33），患者级 balanced accuracy 也从 0.7563 降至 0.7372。因此，2.5D 提高了总体准确率，但默认阈值下漏掉了更多 AD 患者，不能认为它在所有指标上更优。

作为后续验证集阈值探索，将 2.5D 的患者级阈值调至 0.39 时，患者准确率为 80.39%，AD recall 为 78.79%（26/33）；代价是 NC 误报从 7 人增加到 13 人。该阈值是根据验证集选择的，结果不能作为测试集表现。

### English Version

To incorporate spatial information, we replaced three copies of the same grayscale MRI slice with three adjacent slices—the previous, current, and next slices—as the input channels of ConvNeXt-Tiny. The model was trained from scratch using the same patient level split, learning rate (0.001), weight decay (0.0001), and eight-epoch schedule. The checkpoint with the lowest validation loss was obtained at epoch 3 (0.5513).

At the default decision threshold of 0.5, the 2.5D model achieved 74.19% image level, 78.24% scan level, and 79.41% patient level validation accuracy, compared with 73.15%, 76.85%, and 74.51% for the original single-slice ConvNeXt. However, patient level AD recall decreased from 78.79% (26/33) to 57.58% (19/33), while patient level balanced accuracy decreased from 0.7563 to 0.7372. Thus, the gain in overall accuracy came with more missed AD patients at the default threshold.

In an exploratory validation threshold analysis, setting the 2.5D patient level threshold to 0.39 produced 80.39% accuracy and 78.79% AD recall (26/33), while NC false positives increased from 7 to 13. This threshold was selected on the validation set; its performance must not be presented as a test result.

---


## 2. Summary

- The 2.5D input experiment replaces channel duplication with adjacent-slice triplets to incorporate spatial context.
- At default threshold (0.5), overall validation accuracy improved across all levels, but AD recall dropped substantially.
- Threshold tuning on validation data can recover AD recall at the cost of more false positives; such tuning must not be applied to the test set.
- Code changes are concentrated in `dataset.py` (data loading logic); `train.py` and `predict.py` received minor updates for compatibility and logging; `modules.py` is unchanged.

- ## 3. Brightness and Contrast Augmentation Experiment

### 中文版

在 2.5D 输入实验的基础上，我们仅在训练阶段加入小幅随机亮度和对比度变化（各为 ±10%）。相邻切片输入、患者级数据划分、ConvNeXt-Tiny 模型、学习率 0.001、权重衰减 0.0001 和 8 轮训练均保持不变。最低验证损失出现在第 4 轮，为 0.5517；未加入该增强的 2.5D 实验在第 3 轮达到 0.5513。因此，增强后的最佳验证损失没有改善。

在默认阈值 0.5 下，增强后的图像、扫描和患者级验证准确率分别为 **75.74%、78.70% 和 81.37%**，相比未增强的 2.5D 模型的 74.19%、78.24% 和 79.41% 均有所提高。图像级 AD recall 从 62.62% 提升至 64.80%，患者级 AUROC 从 0.8612 提升至 0.8757。然而，患者级 AD recall 仍为 **57.58%（19/33）**，没有改善。这些结果支持保留该增强作为候选方案，但单次实验不足以证明提升一定由增强造成。

### English Version

Building on the 2.5D input experiment, we added mild random brightness and contrast variation (±10% each) during training only. The adjacent-slice input, patient-level split, ConvNeXt-Tiny architecture, learning rate (0.001), weight decay (0.0001), and eight-epoch schedule were unchanged. The lowest validation loss occurred at epoch 4 (0.5517), compared with 0.5513 at epoch 3 without this augmentation. Thus, the best validation loss did not improve.

At the default threshold of 0.5, image-, scan-, and patient-level validation accuracies were **75.74%, 78.70%, and 81.37%**, respectively, compared with 74.19%, 78.24%, and 79.41% for the unaugmented 2.5D model. Image-level AD recall increased from 62.62% to 64.80%, and patient-level AUROC increased from 0.8612 to 0.8757. However, patient-level AD recall remained **57.58% (19/33)**. These results support retaining the augmentation as a candidate setting, although a single run does not establish that the improvement was caused by augmentation.

---

## 4. Patient-Level Threshold Selection and Held-Out Test Results

### 中文版

针对加入增强后的模型，我们使用**验证集患者级预测概率**选择 AD 分类阈值。将患者级阈值从默认的 0.5 降至 0.38 后，验证集患者准确率为 **83.33%（85/102）**，AD recall 为 **78.79%（26/33）**；AD 漏判从 14 人减少至 7 人，而 NC 误报从 5 人增加至 10 人。图像级和扫描级阈值仍保持 0.5。阈值 0.38 随后被冻结，并用于测试集评估。

在测试集上，图像、扫描和患者级准确率分别为 **72.91%、76.44% 和 81.37%**，AUROC 分别为 **0.8135、0.8583 和 0.8555**。患者级混淆矩阵显示，69 名 NC 患者中有 60 名被正确识别，33 名 AD 患者中有 23 名被正确识别；患者级 AD recall 为 **69.70%（23/33）**，仍有 10 名 AD 患者被漏判。

因此，该模型在测试集上达到了预先设定的**患者级准确率和 AUROC 均不低于 0.80** 的目标，但图像级准确率未达到 0.80，AD 漏判也仍是重要限制。阈值是在验证集上选定的；测试集结果仅用于最终评估，不用于再次调整模型或阈值。

### English Version

For the augmented model, we selected the AD decision threshold using **patient-level probabilities from the validation set**. Lowering the patient-level threshold from 0.5 to 0.38 produced **83.33% validation accuracy (85/102)** and **78.79% AD recall (26/33)**. AD false negatives decreased from 14 to 7, while NC false positives increased from 5 to 10. Image- and scan-level thresholds remained at 0.5. The patient-level threshold was then fixed at 0.38 for test evaluation.

On the held-out test set, image-, scan-, and patient-level accuracies were **72.91%, 76.44%, and 81.37%**, with AUROC values of **0.8135, 0.8583, and 0.8555**, respectively. At the patient level, 60 of 69 NC patients and 23 of 33 AD patients were correctly classified. Patient-level AD recall was **69.70% (23/33)**, leaving 10 AD patients incorrectly classified as NC.

The model therefore met the predefined **patient-level accuracy and AUROC targets of at least 0.80** on the test set. Image-level accuracy remained below 0.80, and missed AD patients remain an important limitation. The threshold was selected using validation data; test results were used for final evaluation, not for further model or threshold selection.

## 5. Dropout Regularisation Experiment

### 中文版

为检验分类头正则化能否缓解后期验证损失上升的问题，我们在 ConvNeXt-Tiny 分类头的最终线性层之前加入 `Dropout(p=0.2)`。模型仍从零训练，可训练参数保持为 27,821,666；2.5D 输入、训练阶段的亮度与对比度增强、数据划分、学习率 0.001、权重衰减 0.0001 和 8 轮训练均未改变。因此，本次实验主要比较加入分类头 dropout 前后的结果。

加入 dropout 后，最低验证损失出现在第 2 轮，为 **0.5669**；未加入 dropout 的模型在第 4 轮达到 **0.5517**。按最低验证损失保存的 dropout 模型，图像级验证准确率为 **73.66%**、AUROC 为 **0.8135**，低于未加入 dropout 时的 **75.74%** 和 **0.8257**。扫描级准确率也从 78.70% 降至 75.93%。

在相同的患者级 AD 阈值 **0.38** 下，dropout 模型的患者准确率为 **74.51%**、AUROC 为 **0.8586**，低于未加入 dropout 模型的 **83.33%** 和 **0.8757**。两个模型均正确识别了 33 名 AD 患者中的 26 名，但 NC 误报从 **10/69** 增至 **19/69**。这说明本次患者级准确率下降主要与更多 NC 患者被判为 AD 有关。

Dropout 模型第 2 轮之后的训练准确率继续上升，至第 8 轮达到 85.59%；同期验证损失升至 0.7236。本次设置未能阻止后期验证表现恶化。由于结果只来自一次训练，不能据此断言分类头 dropout 一般无效；但在本项目当前配置下，**没有证据支持保留 `p=0.2` 的 dropout**。因此，后续以未加入 dropout 的 2.5D＋亮度对比度增强模型作为当前候选模型，本次 dropout 模型不进入测试集评估。

### English Version

To examine whether classifier-head regularisation could reduce the later rise in validation loss, we inserted `Dropout(p=0.2)` immediately before the final linear layer of ConvNeXt-Tiny. The model remained randomly initialised with 27,821,666 trainable parameters. The 2.5D input, training-time brightness and contrast augmentation, data split, learning rate (0.001), weight decay (0.0001), and eight-epoch schedule were unchanged. This experiment therefore focused on the effect of adding classifier-head dropout.

With dropout, the lowest validation loss occurred at epoch 2 (**0.5669**), compared with **0.5517** at epoch 4 without dropout. The selected dropout checkpoint achieved **73.66% image-level validation accuracy** and **0.8135 AUROC**, below the corresponding results without dropout (**75.74%** and **0.8257**). Scan-level accuracy also decreased from 78.70% to 75.93%.

Using the same patient-level AD threshold of **0.38**, the dropout model achieved **74.51% patient-level accuracy** and **0.8586 AUROC**, compared with **83.33%** and **0.8757** without dropout. Both models correctly identified 26 of 33 AD patients, but NC false positives increased from **10 of 69** to **19 of 69**. The lower patient-level accuracy was therefore associated mainly with more NC patients being classified as AD.

After epoch 2, training accuracy continued to increase, reaching 85.59% at epoch 8, while validation loss rose to 0.7236. This dropout setting did not prevent later deterioration in validation performance. A single run cannot establish that classifier-head dropout is ineffective in general. However, these results **do not support retaining `p=0.2` dropout in the current configuration**. We retained the 2.5D model with brightness and contrast augmentation but without dropout as the current candidate, and did not evaluate the dropout model on the test set.

---

## 6. Data-Split Clarification

### 中文版

所有上述对比均使用相同的自建患者级划分。当前数据处理代码汇总课程数据原有的训练和测试文件夹后，按患者及类别分层，重新划分为 **70% 训练、15% 验证、15% 测试**，分别包含 **476、102 和 102 名患者**。不同新划分之间没有患者重叠，但这里的"测试集"**不是课程数据原有的测试文件夹**。因此，报告中的测试结果应准确称为"重新划分后保留的测试集结果"；是否还需保持原始测试集独立，应依据作业要求确认。

### English Version

All comparisons above used the same custom patient-level split. The current data preparation code pooled the course dataset's original training and test folders, then created a new stratified **70% training, 15% validation, and 15% test** split by patient, containing **476, 102, and 102 patients**, respectively. Patients do not overlap across the new splits. However, the reported "test set" is **not the course dataset's original test folder**. Test results should therefore be described as results on the newly held-out test split; whether the original test folder must remain independent should be checked against the assignment requirements.
