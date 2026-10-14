# Results — Arabizi Sentiment LoRA Fine-Tuning

**Model:** MBZUAI-Paris/Nile-Chat-4B
**Method:** LoRA (r=8, alpha=16, dropout=0.1) on a 4-bit quantized base model
**Generated:** 2026-07-15T23:47:49.943100+00:00
**Environment:** Kaggle

## Headline results

| Metric | Value |
|---|---|
| Baseline Macro F1 (before LoRA) | 0.7456 |
| Final Macro F1 (after LoRA) | 0.8843 |
| Improvement | +0.1388 |
| Train / Test examples | 1279 / 320 |
| Epochs / Batch size / LR | 5 / 4 / 0.0002 |

## Files in this folder

| File | Description |
|---|---|
| `evaluation_metrics.json` | Final precision/recall/F1 per class, macro & weighted averages, baseline vs. final comparison |
| `classification_report.txt` | Human-readable sklearn classification report with summary header |
| `sample_predictions.csv` | Random sample of 50 test-set predictions (text, true label, predicted label, correct) |
| `misclassified_examples.csv` | Every test example the fine-tuned model got wrong |
| `training_history.csv` | Per-epoch training loss and macro F1, plus the pre-training baseline |
| `confusion_matrix.png` | Confusion matrix heatmap for the final model on the test set |
| `training_curves.png` | Training loss curve and macro F1 curve across epochs |
| `dataset_statistics.json` | Class distribution, train/test split sizes, and text length stats |
| `training_config.json` | Full hyperparameter and run configuration used for this training run |
| `README.md` | This file |

## Reproducing this run

- **Data source:** `/kaggle/input/datasets/nancyelfky/arabizi-dataset2/arabizi_data.csv`
- **Random seed:** 42
- **LoRA adapter saved to:** `/kaggle/working/models/lora_adapter`
