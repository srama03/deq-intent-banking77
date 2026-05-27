import torch
from collections import Counter

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import random
import yaml
import numpy as np

from src.data.load_data import load_banking77
from src.models.baseline_transformer import BaselineModel
from src.models.deq_transformer import DEQModel
from src.models.deq_implicit import DEQImplicitModel

from src.train import collate_batch

"""
> Get top confusion pairs for all three models to compare how "dumb" or reasonable the mistakes are
> helper 1: getting the predicted vals and label for each
> helper 2: getting top_k conf pairs (off diag)
> load the models and apply helpers

"""
@torch.no_grad()
def get_preds_labs(model, dataloader, device):
    # run model on dataloader and return true labels and predictions as lists
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # get logits
        logits = model(input_ids, attention_mask)

        # compute preds= argmax of scores from logits
        preds = logits.argmax(dim=-1)
        
        # convert to list for f1
        labels_cpu = labels.cpu().tolist()
        preds_cpu = preds.cpu().tolist()
        all_labels.extend(labels_cpu)
        all_preds.extend(preds_cpu)
    
    return all_labels, all_preds

def get_top_confusions(all_labels, all_preds, label_names, top_k=10):
    # get the off-diagonal of the confusion matrix {the top k most common mistakes made by the model}
    mistakes = [(label_names[t], label_names[p]) for t, p in zip(all_labels, all_preds) if t!=p]
    counts = Counter(mistakes)
    return counts.most_common(top_k)

def main():
    # loads saved models, gets the top conf pairs and prints them out for each
    with open("configs/baseline.yaml", "r") as f: # for common stuff like device/seed/etc
        cfg = yaml.safe_load(f)
    seed = int(cfg["experiment"]["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cuda" if (cfg["experiment"]["device"]=="auto" and torch.cuda.is_available()) else "cpu"
    print("device: ", device)

    # load data
    train_ds, val_ds, test_ds, label_names = load_banking77(val_ratio=0.1, seed=seed)
    num_labels = len(label_names)
    tok_name = cfg["data"]["tokenizer_name"]
    max_len = int(cfg["data"]["max_len"])
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    vocab_size = tokenizer.vocab_size

    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=lambda x: collate_batch(x,tokenizer,max_len)
    )

    
    # baseline
    ckpt = torch.load("results/runs/baseline_best.pt")
    cfg_b = ckpt["config"]
    # recreate the model using saved config
    model_base = BaselineModel(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=int(cfg_b["model"]["d_model"]),
            n_heads=int(cfg_b["model"]["n_heads"]),
            num_layers=int(cfg_b["model"]["num_layers"]),
            d_ff=int(cfg_b["model"]["d_ff"]),
            dropout=float(cfg_b["model"]["dropout"]),
            num_labels=num_labels)
    # load the weights in
    model_base.load_state_dict(ckpt["model_state_dict"])
    model_base.eval()

    labs_b, preds_b = get_preds_labs(model_base, test_loader, device)
    conf_base = get_top_confusions(labs_b, preds_b, label_names, top_k=10)

    print("\n most common mistakes (baseline):", conf_base)
    
    
    # deq
    ckpt = torch.load("results/runs/deq_transformer_banking77/deq_transformer_best.pt")
    cfg_d = ckpt["config"]
    # recreate the model using saved config
    model_deq = DEQModel(
            vocab_size=vocab_size,
            max_len=max_len,
            tol=float(cfg_d["deq"]["tol"]),
            max_iters_train=int(cfg_d["deq"]["max_iters_train"]),
            max_iters_eval=int(cfg_d["deq"]["max_iters_eval"]),
            alpha=float(cfg_d["deq"]["alpha"]),
            d_model=int(cfg_d["model"]["d_model"]),
            n_heads=int(cfg_d["model"]["n_heads"]),
            d_ff=int(cfg_d["model"]["d_ff"]),
            dropout=float(cfg_d["model"]["dropout"]),
            num_labels=num_labels)
    # load the weights in
    model_deq.load_state_dict(ckpt["model_state_dict"])
    model_deq.eval()
    
    labs_d, preds_d = get_preds_labs(model_deq, test_loader, device)
    conf_deq = get_top_confusions(labs_d, preds_d, label_names, top_k=10)

    print("\n most common mistakes (weight tied):", conf_deq)
    

    # implicit (true deq)
    ckpt = torch.load("results/runs/deq_implicit_banking77/deq_implicit_best.pt")
    cfg_i = ckpt["config"]
    # recreate the model using saved config
    model_imp = DEQImplicitModel(
        vocab_size=vocab_size,
        max_len=max_len,
        tol=float(cfg_i["deq"]["tol"]),
        max_iters_train=int(cfg_i["deq"]["max_iters_train"]),
        max_iters_eval=int(cfg_i["deq"]["max_iters_eval"]),
        alpha=float(cfg_i["deq"]["alpha"]),
        d_model=int(cfg_i["model"]["d_model"]),
        n_heads=int(cfg_i["model"]["n_heads"]),
        d_ff=int(cfg_i["model"]["d_ff"]),
        dropout=float(cfg_i["model"]["dropout"]),
        num_labels=num_labels)
    # load the weights in
    model_imp.load_state_dict(ckpt["model_state_dict"])
    model_imp.eval()

    labs_i, preds_i = get_preds_labs(model_imp, test_loader, device)
    conf_imp = get_top_confusions(labs_i, preds_i, label_names, top_k=10)

    print("\n most common mistakes (implicit deq):", conf_imp)

    
    print("\nDone!")

if __name__ == "__main__":
    main()