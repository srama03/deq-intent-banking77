import os
import math
import yaml
import numpy as np
import random
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.load_data import load_banking77
from src.models.baseline_transformer import BaselineModel

def collate_batch(batch, tokenizer, max_len=64):
    """
    takes list of dicts (batch), extracts text and labels, tokenizes text
    outputs dict of tensors: 
        input_ids: (B, T), attention_mask: (B, T), labels: (B,)
    """
    texts = [ex["text"] for ex in batch]
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors = "pt"
    )
    enc["labels"]=labels
    return enc

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.0
    seen = 0 
    correct = 0
    all_preds = []
    all_labels = []
    # get batch:
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
         
        # get logits
        logits = model(input_ids, attention_mask)
        # get loss
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        correct+= (preds == labels).sum().item()
        # convert to list for f1
        labels_cpu = labels.cpu().tolist()
        preds_cpu = preds.cpu().tolist()
        all_labels.extend(labels_cpu)
        all_preds.extend(preds_cpu)
        # backward pass: get gradients
        loss.backward()
        # update weights
        optimizer.step()
        # reset grads
        optimizer.zero_grad()
        # get batch size, running loss weighted by batch size
        bs = labels.size(0)
        seen += bs
        running_loss +=loss.item()*bs
    train_macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return {"train_loss" : running_loss/max(seen,1),
            "train_acc": correct / max(seen, 1),
            "train_macro_f1": train_macro_f1,}


@torch.no_grad()
def eval_one_epoch(model, dataloader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    seen = 0
    correct = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # get logits
        logits = model(input_ids, attention_mask)
        # get loss
        loss = loss_fn(logits, labels)
        # get running loss, weight by batch size
        bs = labels.size(0)
        running_loss += loss.item()*bs
        # compute preds= argmax of scores from logits
        preds = logits.argmax(dim=-1)
        # compute num of seen batches and num of correct preds
        seen+=bs
        correct+= (preds == labels).sum().item()
        # convert to list for f1
        labels_cpu = labels.cpu().tolist()
        preds_cpu = preds.cpu().tolist()
        all_labels.extend(labels_cpu)
        all_preds.extend(preds_cpu)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return {
        "val_loss": running_loss / max(seen, 1),
        "val_acc": correct / max(seen, 1),
        "val_macro_f1": macro_f1
    }


def main(config_path="configs/baseline.yaml"):
    # configs
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg["experiment"]["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cuda" if (cfg["experiment"]["device"]=="auto" and torch.cuda.is_available()) else "cpu"
    print("device: ", device)

    # data
    train_ds, val_ds, test_ds, label_names = load_banking77(val_ratio=0.1, seed=seed)
    num_labels = len(label_names)
    tok_name = cfg["data"]["tokenizer_name"]
    max_len = int(cfg["data"]["max_len"])
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        collate_fn=lambda x: collate_batch(x,tokenizer,max_len)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=lambda x: collate_batch(x,tokenizer,max_len)
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=lambda x: collate_batch(x,tokenizer,max_len)
    )

    # model
    vocab_size = tokenizer.vocab_size
    model = BaselineModel(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        num_layers=int(cfg["model"]["num_layers"]),
        d_ff=int(cfg["model"]["d_ff"]),
        dropout=float(cfg["model"]["dropout"]),
        num_labels=num_labels).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    # train
    epochs = int(cfg["train"]["epochs"])
    #epochs = 2
    run_dir = os.path.join(cfg["experiment"]["output_dir"], cfg["experiment"]["name"]) # folder to save results/runs
    os.makedirs(run_dir, exist_ok=True) # creates said folder if it doesnt alr exist
    best_path = os.path.join(run_dir, "baseline_best.pt") # file name for the best model

    best_f1 = float("-inf")
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = eval_one_epoch(model, val_loader, device)
        print("train:", train_stats)
        print("val:", val_stats)
        current_f1 = val_stats["val_macro_f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "model_state_dict": model.state_dict(), # learned weights
                    "config": cfg,
                },
                best_path
            )
            print(f"Best checkpoint: epoch={best_epoch}, val_macro_f1={best_f1:.4f}")
    print(f"\nBest checkpoint: epoch={best_epoch}, val_macro_f1={best_f1:.4f}")
    print(f"Saved to: {best_path}")
    
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    # testing timeeee
    test_stats = eval_one_epoch(model, test_loader, device)
    test_stats = {k.replace("val_", "test_"): v for k, v in test_stats.items()}
    print("test:", test_stats)


    print("\nDone.")

if __name__ == "__main__":
    main()