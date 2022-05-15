import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, accuracy_score


def save(model, output_path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, output_path)


def calc_mae(logits, labels):
    softmax = F.softmax(logits, dim=1)
    float_predict = softmax@torch.tensor([0, 1, -1]).numpy()
    int_predict = softmax.argmax(dim=1).numpy()
    int_predict = np.where(int_predict == 2, -1, int_predict)

    labels[labels == 2] = -1
    mae_float = mean_absolute_error(float_predict, labels)
    mae_int = mean_absolute_error(int_predict, labels)
    return mae_float, mae_int


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def calc_accuracy(pred_logits, true_labels):
    pred_labels = F.softmax(pred_logits, dim=1).argmax(dim=1)
    return accuracy_score(pred_labels, true_labels)
