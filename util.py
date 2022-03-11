from tracemalloc import start
import torch
import torch.nn.functional as F
import shutil
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp

def get_data_dir():
    return osp.join(osp.dirname(osp.realpath(__file__)), "data")

def get_dataset_dir():
    return osp.join(osp.dirname(osp.realpath(__file__)), "data/geomat")


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


def run_training(model_name, train, test, model, optimizer, scheduler, total_epochs):
    last_checkpoint = f"data/checkpoints/{model_name}_checkpoint.pt"
    if os.path.isfile(last_checkpoint):
        model, optimizer, start_epoch = load_ckp(last_checkpoint, model, optimizer, scheduler)
    else:
        start_epoch = 1
    writer = get_writer(model_name)
    best_test_acc = 0
    print(f"Starting at epoch {start_epoch}")
    for epoch in range(start_epoch, total_epochs):
        loss, train_acc, balanced_train_acc = train()
        test_acc = test()
        print(f"Epoch {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Balanced Train Acc: {balanced_train_acc:.4f}, Test: {test_acc:.4f}")
        scheduler.step()

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Balanced_Accuracy/train", balanced_train_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        checkpoint = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
        save_ckp(checkpoint, test_acc >= best_test_acc, "data/checkpoints", model_name)

        best_test_acc = max(test_acc, best_test_acc)

    writer.close()
