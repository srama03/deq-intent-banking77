import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import random
import yaml
import numpy as np

from src.data.load_data import load_banking77
from src.models.baseline_transformer import BaselineModel
from src.models.deq_transformer import DEQModel
from src.models.deq_implicit import DEQImplicitModel

from src.train import collate_batch, eval_one_epoch
from src.data.noise import add_noise

def main():
    """
    loads the saved models and evaluates each on both the clean and noisy test dataset
    """
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

    # create noisy version of test set for robustness eval
    noisy_test_ds = add_noise(test_ds, noise_level=0.1)

    # load noisy
    noisy_test_loader = DataLoader(
        noisy_test_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=lambda x: collate_batch(x, tokenizer, max_len)
    )

    # baseline
    ckpt = torch.load("results/runs/baseline_best.pt")
    cfg_b = ckpt["config"]
    # recreate the model using saved config
    model = BaselineModel(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=int(cfg_b["model"]["d_model"]),
            n_heads=int(cfg_b["model"]["n_heads"]),
            num_layers=int(cfg_b["model"]["num_layers"]),
            d_ff=int(cfg_b["model"]["d_ff"]),
            dropout=float(cfg_b["model"]["dropout"]),
            num_labels=num_labels)
    # load the weights in
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # test on clean
    test_base = eval_one_epoch(model, test_loader, device)
    test_base = {k.replace("val_", "test_"): v for k, v in test_base.items()}
    print("\nbaseline clean:", test_base)
    # test on noisy
    test_stats_base = eval_one_epoch(model, noisy_test_loader, device)
    test_stats_base = {k.replace("val_", "test_"): v for k, v in test_stats_base.items()}
    print("\nbaseline noisy:", test_stats_base)
    


    # deq
    ckpt = torch.load("results/runs/deq_transformer_banking77/deq_transformer_best.pt")
    cfg_d = ckpt["config"]
    # recreate the model using saved config
    model = DEQModel(
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # test on clean
    test_deq = eval_one_epoch(model, test_loader, device)
    test_deq = {k.replace("val_", "test_"): v for k, v in test_deq.items()}
    print("\ndeq clean:", test_deq)
    # test on noisy
    test_stats_deq = eval_one_epoch(model, noisy_test_loader, device)
    test_stats_deq = {k.replace("val_", "test_"): v for k, v in test_stats_deq.items()}
    print("\ndeq noisy:", test_stats_deq)

    # implicit (true deq)
    ckpt = torch.load("results/runs/deq_implicit_banking77/deq_implicit_best.pt")
    cfg_i = ckpt["config"]
    # recreate the model using saved config
    model = DEQImplicitModel(
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # test on clean
    test_imp = eval_one_epoch(model, test_loader, device)
    test_imp = {k.replace("val_", "test_"): v for k, v in test_imp.items()}
    print("\ntrue deq (implicit) clean:", test_imp)
    # test on noisy
    test_stats_imp = eval_one_epoch(model, noisy_test_loader, device)
    test_stats_imp = {k.replace("val_", "test_"): v for k, v in test_stats_imp.items()}
    print("\ntrue deq (implicit) noisy:", test_stats_imp)

    print("\n--- RESULTS SUMMARY ---")
    print(f"{'Model':<20} {'Clean F1':>10} {'Noisy F1':>10} {'Drop':>10}")
    print(f"{'Baseline':<20} {test_base['test_macro_f1']:>10.4f} {test_stats_base['test_macro_f1']:>10.4f} {test_base['test_macro_f1']-test_stats_base['test_macro_f1']:>10.4f}")
    print(f"{'DEQ-like':<20} {test_deq['test_macro_f1']:>10.4f} {test_stats_deq['test_macro_f1']:>10.4f} {test_deq['test_macro_f1']-test_stats_deq['test_macro_f1']:>10.4f}")
    print(f"{'True DEQ':<20} {test_imp['test_macro_f1']:>10.4f} {test_stats_imp['test_macro_f1']:>10.4f} {test_imp['test_macro_f1']-test_stats_imp['test_macro_f1']:>10.4f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
