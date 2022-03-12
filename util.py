from tracemalloc import start
import torch
import torch.nn.functional as F
import shutil
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp
import h5py
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import textwrap
import json
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def get_data_dir():
    return osp.join(osp.dirname(osp.realpath(__file__)), "data")

def get_dataset_dir():
    return osp.join(osp.dirname(osp.realpath(__file__)), "data/geomat")

class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


def save_h5_features(fname_h5_feat2d, data_key, data_value):
    h5_file = h5py.File(fname_h5_feat2d.strip(), "a")
    try:
        del h5_file[data_key]
    except:
        pass
    h5_file.create_dataset(data_key, data=data_value, compression="gzip", compression_opts=4, dtype="float")
    h5_file.close()


def save_ckp(state, is_best, checkpoint_dir, model_name):
    f_path = f"{checkpoint_dir}/{model_name}_checkpoint.pt"
    torch.save(state, f_path)
    if is_best:
        shutil.copyfile(f_path, f"{checkpoint_dir}/{model_name}_best_model.pt")


def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, checkpoint["epoch"]

def criterion(pred, gold, smoothing=True):
    """Calculate cross entropy loss, apply label smoothing if needed."""

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction="mean")

    return loss


def get_writer(model_name):
    return SummaryWriter(log_dir=f"runs/{model_name}")


def run_training(model_name, train, test, model, optimizer, scheduler, total_epochs, cm=False):
    best_checkpoint = f"data/checkpoints/{model_name}_best_model.pt"
    if os.path.isfile(best_checkpoint):
        best_test_acc = torch.load(best_checkpoint)["test_acc"]
    else:
        best_test_acc = 0

    last_checkpoint = f"data/checkpoints/{model_name}_checkpoint.pt"
    if os.path.isfile(last_checkpoint):
        model, optimizer, start_epoch = load_ckp(last_checkpoint, model, optimizer, scheduler)
    else:
        start_epoch = 1
    writer = get_writer(model_name)
    
    print(f"Starting at epoch {start_epoch}")
    for epoch in range(start_epoch, total_epochs):
        if cm:
            loss, train_acc, balanced_train_acc, train_cm = train()
            test_acc, test_cm = test()
            writer.add_figure('Confusion_Matrix/train', train_cm.plot(normalize=True), epoch)
            writer.add_figure('Confusion_Matrix/test', test_cm.plot(normalize=True), epoch)
        else:
            loss, train_acc, balanced_train_acc = train()
            test_acc = test()
            
        print(f"Epoch {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Balanced Train Acc: {balanced_train_acc:.4f}, Test: {test_acc:.4f}")
        scheduler.step()

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Balanced_Accuracy/train", balanced_train_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        checkpoint = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "test_acc": test_acc}
        save_ckp(checkpoint, test_acc >= best_test_acc, "data/checkpoints", model_name)

        best_test_acc = max(test_acc, best_test_acc)

    print(f"Finished training at epoch {total_epochs - 1}")

    writer.close()


class ConfusionMatrixMeter():
    def __init__(self, labels, cmap='orange'):
        self._cmap = cmap
        self._k = len(labels)
        self._labels = labels
        self._cm = np.ndarray((self._k, self._k), dtype=np.int32)
        self.reset()

    def reset(self):
        self._cm.fill(0)

    def add(self, target, predicted):

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'
        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self._k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self._k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'
        self._cm += confusion_matrix(target, predicted, labels=range(0, self._k))

    def value(self, normalize=False):
        if normalize:
            np.set_printoptions(precision=2)
            return np.divide(self._cm.astype('float'), self._cm.sum(axis=1).clip(min=1e-12)[:, np.newaxis])
        else:
            return self._cm

    def accuracy(self):
        return np.divide(self.value().trace(), self.value().sum())*100

    def mean_acc(self):
        return np.divide(self.value(True).trace(), self._k)*100

    def save_json(self, filename, normalize=False):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding="utf8") as f:
            json.dump(self.value(normalize=normalize).tolist(), f)

    def save_npy(self, filename, normalize=False):
        np.save(filename, self.value())

    def plot(self, normalize=False):
        cm = self.value(normalize=normalize)
        fig = plt.figure(figsize=(self._k, self._k), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        if normalize:
            cm_plot = ax.imshow(cm, cmap=self._cmap, vmin=0, vmax=1)
        else:
            cm_plot = ax.imshow(cm, cmap=self._cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.figure.colorbar(cm_plot, cax=cax)
        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in self._labels]
        classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]
        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted')
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(self._k), range(self._k)):
            if normalize:
                ax.text(j, i, np.round(cm[i, j], 2), horizontalalignment="center",
                        verticalalignment='center', color="black")
            else:
                ax.text(j, i, int(cm[i, j]) if cm[i, j] != 0 else '.',
                        horizontalalignment="center", verticalalignment='center', color="black")
        fig.set_tight_layout(True)
        return fig
