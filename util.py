import torch
import torch.nn.functional as F
import shutil


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
