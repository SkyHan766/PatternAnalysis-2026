# ADNI MRI Slice Classification with a Compact ConvNeXt

**COMP3710 Pattern Recognition — final project report**

**Author:** Jinqiao Han | **Model:** ConvNeXt, hard difficulty | **Date:** 27 October 2026

## Project overview

This report evaluates Alzheimer’s disease versus cognitively normal classification from individual ADNI MRI slices. The final assessed model is a compact, randomly initialized ConvNeXt trained on patient-disjoint data. It reached **76.02% accuracy on 4,500 held-out test slices**, below the course’s 80% slice-level requirement, while the supplementary patient-level accuracy was **80.39%**. We document model selection, comparisons, overfitting, confident errors and the limits of these results.

| Report component | Location |
|---|---|
| Problem, feasibility and experimental protocol | Body page 1 |
| Experiments and measured test performance | Body page 2 |
| Confidence, error interpretation and conclusion | Body page 3 |
| Training curves and four test-set error images | Visual appendix |

The feasibility pilot using ImageNet weights reached 86.04% test slice accuracy, but the later tutor requirement excluded pretrained weights. Its result is discussed as exploratory context; **the final course comparison uses training from scratch**.

<!-- PAGE BREAK -->

## 1. Problem, feasibility and experimental protocol

The prototype classifies individual 2D ADNI MRI slices as Alzheimer's disease (AD) or cognitively normal (CN, labelled NC in code). The course target is 80% test **slice-level** accuracy; scan- and patient-level figures are supplementary aggregations, not substitutes. The engineering question is whether the additional complexity of ConvNeXt improves reliability over a basic CNN, especially for ambiguous images and confidently wrong predictions. This is a research prototype, not a clinical diagnosis system.

**Feasibility and changed target.** The submitted feasibility document identified researchers as intended users and specified patient-disjoint splits, reporting image and patient metrics, and test patient accuracy and AUROC of at least 0.80 after an eight-epoch pretrained pilot. That pilot attained 86.04% test slice accuracy but later tutor guidance required random initialization. The tutor also clarified that the graded 80% threshold applies to **individual test slices**. The subsequent compliant experiments therefore used weights initialized from scratch; the initial patient-level targets remain supplementary. The feasibility document anticipated overfitting and suggested training changes and a fallback; a smaller network and CNN comparison were subsequently investigated.

The audited corpus has 30,520 slices, 1,526 scans and 680 subjects. For the **primary evaluated checkpoint**, the manifest contains 21,700 training, 4,320 validation and 4,500 test slices; subjects are disjoint across splits (476/102/102). Splitting by subject keeps different scans and slices of one person together. One scan contributes 20 slices, so the nominal number of images greatly overstates the number of independent people. Images are resized/cropped to 224 pixels and normalized. Training-only random transforms include modest geometric changes and brightness/contrast jitter; evaluation is deterministic. The dataset has a possible label/source-time association in raw filenames (older-looking timestamps appear only with AD in the inspected subset); filename time has not been verified as acquisition date. This is a possible confound, not an established cause of errors.

The principal model is a **3,178,018-parameter, one-channel ConvNeXt variant** with widths 32/64/128/256, depths 3/3/9/3, depthwise convolution, normalization, pointwise channel mixing and stochastic depth up to 0.2. It is trained from scratch. For the archived run with complete test predictions: AdamW, learning rate 0.001, weight decay 0, batch size 64, 50 epochs, seed 42, with mild intensity/geometric augmentation; no cosine scheduler or label smoothing. A checkpoint is selected from epochs within 0.005 of the highest validation image accuracy, taking the lowest validation loss. Despite its folder name containing “e20” and “lr3e_4”, the saved configuration states **50 epochs and learning rate 0.001**. Archived PyTorch version: 2.11.0+cu130; this is the version logged for this run, not necessarily every experiment.

<!-- PAGE BREAK -->

## 2. Experimental progression and quantitative findings

| Investigation | Rationale and actual observation |
|---|---|
| Full-width ConvNeXt (27.82M) | From scratch, LR 0.001, WD 0.0001: the 8-epoch minimum-loss checkpoint was epoch 2 (73.15% validation accuracy). At 30 epochs the selected checkpoint was unchanged; final training accuracy reached 98.18%. |
| Baseline CNN (27.88M) | The matched eight-epoch README run selected epoch 7: **75.88% validation** slice accuracy versus ConvNeXt's 73.15% at its selected epoch. Baseline test results were not supplied. |
| Pretrained pilot (not compliant) | ImageNet-pretrained Tiny, two head-only and six fine-tuning epochs: **86.04% test slice**, 89.22% test patient accuracy. Excluded from the required scratch-trained result. |
| 2.5D and intensity jitter | Adjacent slices replaced duplicated channels: validation slice accuracy 73.15% → 74.19%; adding ±10% brightness/contrast jitter reached 75.74% and 81.37% patient validation accuracy. These are earlier configuration results. |
| Regularization and capacity | Classifier dropout 0.2 reduced selected validation accuracy to 73.66%. Narrow ResNet-18 (2.80M) and ConvNeXt (3.18M) still overfit. LR, WD, warm-up/cosine, label smoothing, stochastic depth and checkpoint rules were explored; effects are configuration-dependent. |
| One-channel 50-epoch run | Selected epoch 22: 75.90% validation slice accuracy and **76.02% test (3,421/4,500)**. Training accuracy rose from 93.82% at epoch 22 to 98.12% at epoch 50 while final validation accuracy was 74.47%. Training metrics include stochastic augmentation. |

**Held-out test results of the primary checkpoint (threshold 0.5):**

| Level | N | Accuracy | Balanced acc. | AD recall | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Slice (course target) | 4,500 | **76.02%** | 75.64% | 66.11% | 0.7564 | 0.8536 |
| Scan (probabilities averaged) | 225 | 80.44% | 80.06% | 70.37% | 0.8011 | 0.9126 |
| Patient (scan probabilities averaged) | 102 | 80.39% | 76.02% | 63.64% | 0.7683 | 0.9007 |

The slice-level confusion matrix (rows actual CN/AD, columns predicted CN/AD) is **[[1993, 347], [732, 1428]]**. Hence the model misses 732 of 2,160 AD slices. Patient accuracy exceeds 80%, but 12 of 33 AD patients remain misclassified; the patient result does not meet the slice-level target. The selected run used 0.19 GB peak GPU memory during test inference and 0.867 ms/slice including data loading. Its parameter count is roughly 89% below the original 27.82M model; the entire reduction must not be attributed to changing three input channels to one.

<!-- PAGE BREAK -->

## 3. Confidence, failure autopsy and engineering decision

Using `max(P(AD), 1-P(AD))` as confidence, **590/3,409** test slices accepted at a 0.90 cutoff are wrong, including **438 AD-to-CN errors**. Accepted accuracy is 82.69% at 75.76% coverage, **not** the full test accuracy. This cutoff was explored on test predictions and is not a valid preselected deployment rule. Any future reject option must be selected on validation data and report coverage and AD misses; probability calibration remains unproven.

Appendix Figures A1–A2 show the training trajectory and four distinct held-out subject errors. Two mistakes have probabilities extremely close to zero or one for the wrong class; two lie near the 0.5 boundary. Without expert review, no specific anatomical, acquisition or biological cause can be assigned to a slice. Confidence does not guarantee correctness, and a referral rule needs a measured AD-miss rate; clinical-use claims are unsupported.

**Why the changes were tried.** Adjacent slices add local anatomical context; brightness and contrast jitter encourage robustness to modest intensity changes. Weight decay, dropout and stochastic depth aim to limit reliance on training-specific details. Warm-up/cosine schedules vary optimization step size, while EMA averages model weights over updates. In these runs, lowering the full-width model's fixed LR from 0.001 to 0.0003 changed selected validation slice accuracy only from 73.15% to 73.10%. A separate one-channel WD 0.01 run reached 75.29% test slice accuracy versus 76.02% for the WD 0 run; its optimization details differed, so the gap should not be attributed solely to WD.

**Subsequent experiment and limits.** On a different patient-disjoint 70/10/20 split (20,800/3,080/6,640 slices), a 60-epoch run used LR 3e-4, WD 0.05, cosine decay, label smoothing 0.1 and EMA 0.999. EMA validation balanced accuracy peaked at 76.85% (epoch 22); the supplied archive lacks matching test predictions. An earlier non-EMA run on that split achieved 77.40% validation and 73.04% test slice accuracy. Different splits prevent direct ranking, and EMA's test benefit is unknown. A later 70/20/10 split has no measured result here.

**Conclusion and reproducibility.** The compliant scratch-trained prototype works, but its **76.02% test slice accuracy misses the 80% course target**. It meets the original feasibility document's supplementary patient goals (80.39% accuracy; AUROC 0.9007). Preserve the manifest, seed, exact checkpoint and configuration. A same-split baseline test, validation-derived confidence rule and expert review of failures remain useful next steps. Full dependency versions and a baseline test score were not supplied, so full reproducibility is not yet established.

**AI disclosure.** ChatGPT and Kimi assisted with coding, debugging, planning and interpretation. Numerical claims were cross-checked against supplied logs, configurations and predictions; the author is responsible for this work.

**References.** Liu et al., “A ConvNet for the 2020s,” CVPR 2022. Loshchilov and Hutter, “Decoupled Weight Decay Regularization,” ICLR 2019, arXiv:1711.05101. COMP3710 Report: Pattern Recognition, v2.0 draft, 2026. Han, Feasibility of ConvNeXt and project README, 2026; experiment logs and saved predictions.

<!-- PAGE BREAK -->

## Visual appendix

**Figure A1.** Accuracy and cross-entropy loss from the primary 50-epoch run. The selected checkpoint came from epoch 22. The train curve was computed during stochastic training, whereas validation used deterministic inference; the persistent divergence is consistent with limited generalization.

![Figure A1. Training and validation accuracy and loss across 50 epochs.](COMP3710_training_curves.png)

**Figure A2.** Four held-out slices misclassified by the selected model; numbers correspond to the table below. Images are displayed without intensity adjustment. Predicted values are probabilities assigned to AD, not independently calibrated probabilities of disease.

![Figure A2. Four misclassified MRI slices from distinct held-out subjects.](COMP3710_failure_cases.png)

| Case: actual → predicted | Subject and slice | P(AD) |
|---|---|---:|
| 1: CN → AD | 023_S_1190; `1251421_104.jpeg` | 0.9998 |
| 2: AD → CN | 127_S_5028; `375181_98.jpeg` | <0.0001 |
| 3: CN → AD | 003_S_6257; `974338_99.jpeg` | 0.5005 |
| 4: AD → CN | 130_S_4984; `342274_94.jpeg` | approximately 0.5000 |
