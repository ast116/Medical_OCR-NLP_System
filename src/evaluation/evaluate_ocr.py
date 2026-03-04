import os
from src.evaluation.ocr_metrics import cer, wer
from .medical_metrics import medical_cer

def evaluate_folder(gt_dir, ocr_dir):
    total_cer = []
    total_wer = []
    total_medical_cer = []


    for file in os.listdir(gt_dir):
        gt_path = os.path.join(gt_dir, file)
        ocr_path = os.path.join(ocr_dir, file)

        if not os.path.exists(ocr_path):
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt = f.read()

        with open(ocr_path, "r", encoding="utf-8") as f:
            ocr = f.read()

        total_cer.append(cer(gt, ocr))
        total_wer.append(wer(gt, ocr))
        total_medical_cer.append(medical_cer(gt, ocr))

    return {
        "CER": sum(total_cer) / len(total_cer),
        "WER": sum(total_wer) / len(total_wer),
        "Medical CER": sum(total_medical_cer) / len(total_medical_cer),
    }