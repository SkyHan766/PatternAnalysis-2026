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
