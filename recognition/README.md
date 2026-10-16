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
