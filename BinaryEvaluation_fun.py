from monai.metrics import DiceMetric,SurfaceDiceMetric,HausdorffDistanceMetric
import csv
import SimpleITK as sitk
import os
import torch
import matplotlib.pyplot as plt



def metrics_fun_v1(targetPet, itv):
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
    dice_metric(y_pred=targetPet, y=itv)
    dice1 = dice_metric.aggregate().item()
    print('Dice:', dice1)
    dice_metric.reset()

    hausdorff_metric(y_pred=targetPet, y=itv)
    hausd1 = hausdorff_metric.aggregate().item()
    hausdorff_metric.reset()
    print('Hausdorff:', hausd1)

    return dice1, hausd1


def saveMetrics_fun(path_to_save, file_to_save, all_metric_rows):
    print("Saving Metrics for Rows: ", len(all_metric_rows))
    f = open(path_to_save + file_to_save, 'w', encoding='UTF8')
    writer = csv.writer(f)
    writer.writerow(['HausD', 'Dice'])
    for i in range(len(all_metric_rows)):
        writer.writerows(all_metric_rows[i])
    f.close()
    return 0