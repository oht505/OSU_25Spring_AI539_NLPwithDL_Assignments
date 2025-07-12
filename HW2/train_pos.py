import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse
import numpy as np

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import matplotlib.pyplot as plt
import datetime
import random
import string
import wandb
from tqdm import tqdm

# Import our own files
from datasets.PoSData import Vocab, getUDPOSDataloaders
from models.PoSGRU import PoSGRU
import gensim.downloader

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

config = {
    "bs": 256,  # batch size
    "lr": 0.0005,  # learning rate
    "l2reg": 0.0000001,  # weight decay
    "max_epoch": 30,
    "layers": 2,
    "embed_dim": 100,
    "hidden_dim": 256,
    "residual": True,
    "use_glove": True
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument("--use_glove", type=lambda x: x.lower() == "true", default=config["use_glove"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--layers", type=int, default=config["layers"])
args, _ = parser.parse_known_args()
config["use_glove"] = args.use_glove
config["seed"] = args.seed
config["layers"] = args.layers
set_seed(config["seed"])


def main():
    # Get dataloaders
    train_loader, val_loader, _, vocab = getUDPOSDataloaders(config["bs"])

    vocab_size = vocab.lenWords()
    label_size = vocab.lenLabels()

    ##################################
    #  Q11
    ##################################
    # Preload GloVE vectors
    if config['use_glove']:
        config['embed_dim'] = 100
        embed_init = torch.randn(vocab_size, config['embed_dim'])
        glove_wv = gensim.downloader.load('glove-wiki-gigaword-100')
        count = 0
        missing_words = []
        for i, w in vocab.idx2word.items():
            if w in glove_wv:
                embed_init[i, :] = torch.from_numpy(glove_wv[w])
                count += 1
            else:
                missing_words.append(w)
        print("Word vectors loaded: {} / {}".format(count, vocab_size))
        print(f"Example missing words: {missing_words[:3]}")
    else:
        embed_init = None

    # Build model
    model = PoSGRU(vocab_size=vocab_size,
                   embed_dim=config["embed_dim"],
                   hidden_dim=config["hidden_dim"],
                   num_layers=config["layers"],
                   output_dim=label_size,
                   residual=config["residual"],
                   embed_init=embed_init)
    print(model)

    # Start model training
    train(model, train_loader, val_loader)


############################################
# Skeleton Code
############################################

def train(model, train_loader, val_loader):
    # Log our exact model architecture string
    config["arch"] = str(model)
    run_name = generateRunName()

    # Startup wandb logging
    wandb.login()
    wandb.init(project="[AI539] UDPOS HW2", name=run_name, config=config)

    # Move model to the GPU
    model.to(device)

    ##################################
    #  Q6
    ##################################
    # Set up optimizer and our learning rate schedulers
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])  # TODO
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.25, total_iters=int(config["max_epoch"] * 0.1)),
            CosineAnnealingLR(optimizer, T_max=config["max_epoch"] - int(config["max_epoch"] * 0.1))
        ],
        milestones=[int(config["max_epoch"] * 0.1)]
    )  # TODO

    ##################################
    #  Q7
    ##################################
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # TODO

    # Main training loop with progress bar
    iteration = 0
    pbar = tqdm(total=config["max_epoch"] * len(train_loader), desc="Training Iterations", unit="batch")

    # For safety
    with open("vocab.pkl", "wb") as f:
        pickle.dump(train_loader.dataset.vocab, f)

    for epoch in range(config["max_epoch"]):

        # Q9
        wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

        model.train()

        for x, y, lens in train_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            ##################################
            #  Q7
            ##################################
            loss = criterion(out.view(-1, out.shape[-1]), y.view(-1))  # TODO

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            nonpad = (y != -1).to(dtype=float).sum().item()
            acc = (torch.argmax(out, dim=2) == y).to(dtype=float).sum() / nonpad

            pbar.update(1)

            # Q9
            wandb.log({"Loss/train": loss.item(), "Acc/train": acc.item()}, step=iteration)

            iteration += 1

        val_loss, val_acc = evaluate(model, val_loader)

        # Q9
        wandb.log({"Loss/val": val_loss, "Acc/val": val_acc}, step=iteration)

        ##################################
        #  Q8
        ##################################
        # TODO
        best_val_acc = 0.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

            with open("vocab.pkl", "wb") as f:
                pickle.dump(train_loader.dataset.vocab, f)

        # Adjust LR
        scheduler.step()

    wandb.finish()
    pbar.close()


##################################
#  Q8
##################################
def evaluate(model, loader):
    model.eval()
    # TODO
    total_loss = 0.0
    total_correct = 0.0
    total_nonpad = 0.0

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    with torch.no_grad():
        for x, y, lens in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = criterion(out.view(-1, out.shape[-1]), y.view(-1))
            total_loss += loss.item() * x.size(0)

            nonpad = (y != -1).to(dtype=float).sum().item()
            acc = (torch.argmax(out, dim=2) == y).to(dtype=float).sum().item()

            total_correct += acc
            total_nonpad += nonpad

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / total_nonpad
    return avg_loss, avg_acc


def generateRunName():
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    now = datetime.datetime.now()
    run_name = "" + random_string + "_UDPOS"
    return run_name


if __name__ == "__main__":
    main()
