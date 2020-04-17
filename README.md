# Predicting Patient Discharge


### Abstract
Introduction: The Risk Assessment Prediction Tool (RAPT), a 6-question survey which predicts discharge disposition following elective total joint arthroplasty (TJA), has been shown to be accurate for approximately 75% of cases. We evaluated if accurate discharge predictions can be achieved using basic electronic health record (EHR) data combined with machine learning (ML) algorithms.

Methods: Three models were developed. Model 1 (M1) evaluated the accuracy of predicted discharge disposition in concordance with the published RAPT protocol. Model 2 (M2) utilized the RAPT questions and implemented ML. Model 3 (M3) was developed with non-RAPT EHR data (age, discharge preference and surgeon) and the same ML parameters as M2. Evaluation metrics included overall accuracy for home discharge (HD) positive predictive value (PPV), negative predictive value (NPV), sensitivity, specificity and area under the receiver operator curves (AUROCs).

Results: In total, 1,405 patients with complete RAPT scores were included. The training and test set sizes were 1,124 and 281 patients, respectively. When applied to the test set, the overall accuracy for home discharge of M1 was 83.57%, PPV was 92.14%, NPV was 45.09%, sensitivity was 0.8828 and specificity was 0.5610. When using M2, the overall accuracy decreased to 82.86%, PPV to 91.70%, NPV to 43.14%, sensitivity to 0.8786, specificity to 0.5366 and mean AUROC of 0.87±0.03. For M3, overall accuracy increased to 90.36%, PPV to 95.20%, NPV to 68.63%, sensitivity to 0.9316, specificity to 0.7609 and AUROC to 0.91±0.02.

Conclusion: Utilization of basic EHR data, combined with Extreme Gradient Boosting (XGBoost), can exceed the accuracy of previous generation discharge disposition tools such as the RAPT questionnaire.
