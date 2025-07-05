# 
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def categorical_scorer(
        target: pd.Series,
        y_pred: pd.Series,
        threshold: float = 0.5,
        ):
        """
        For all models:
        input: (target, y_pred)
        output: accuracy, precision, recall, f1_score
        """
        #print(f'threshold: {threshold}')
        y_pred_binary = (y_pred >= threshold).astype(int)

        TN = np.sum((target == 0) & (y_pred_binary == 0))
        TP = np.sum((target == 1) & (y_pred_binary == 1))
        FN = np.sum((target == 1) & (y_pred_binary == 0))
        FP = np.sum((target == 0) & (y_pred_binary == 1))

        #print(f'TN: {TN}, TP: {TP}, FN: {FN}, FP: {FP}')

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 / ((1 / precision) + (1 / recall)) if (precision + recall) > 0 else 0

        return accuracy, precision, recall, f1
