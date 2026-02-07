from datasets import load_dataset
import numpy as np

def load_banking77(val_ratio=0.1, seed=42):
    """
    loads the dataset from hugging face datasets
    returns:
        train_ds
        test_ds
        val_ds
        label_names-> list[str]; maps label id to label name
    """
    ds = load_dataset("banking77")
    full_train = ds["train"]
    test_ds = ds["test"]
    split = full_train.train_test_split(test_size=val_ratio, seed=seed, shuffle=True)
    train_ds = split["train"]
    val_ds = split["test"]

    # get label_names from full_train.features["label"]
    label_names = full_train.features["label"].names
    
    return train_ds, val_ds, test_ds, label_names


def get_label_stats(train_ds):
    """
    Computes simple label distribution stats on the training split.
    """
    labs = np.array(train_ds["label"])
    label_names = train_ds.features["label"].names
    labels, count = np.unique(labs, return_counts=True)

    #  min_count, max_count
    sorted_desc = np.argsort(count)[::-1]
    min_count = count[sorted_desc][-1]
    max_count = count[sorted_desc][0]
    # compute top5 and last5 lists of (labels, label_name, count)
    top5_labels = labels[sorted_desc][:5]
    last5_labels = labels[sorted_desc][-5:]
    top5_label_names = [label_names[i] for i in top5_labels]
    last5_label_names = [label_names[i] for i in last5_labels]
    top5_counts = count[sorted_desc][:5]
    last5_counts = count[sorted_desc][-5:]
    stats = {
        "num_labels": len(label_names),
        "min_count": min_count,
        "max_count": max_count,
        "top5": list(zip(top5_labels.tolist(), top5_label_names, top5_counts.tolist())),
        "bottom5": list(zip(last5_labels.tolist(),last5_label_names , last5_counts.tolist()))
    }
    return stats

def get_token_length_stats(dataset, tokenizer, text_col="text", max_len=64):
    """
    Computes token length percentiles and truncation rate for a given dataset split.
    """
    texts = dataset[text_col]
    if not isinstance(texts, list):
        texts = list(texts)
    enc = tokenizer( texts,
        add_special_tokens=True,
        truncation=False)
    lens = np.array([len(ids) for ids in enc["input_ids"]])
    p50, p90, p95, p99 = np.percentile(lens, [50, 90, 95, 99])
    stats = {
        "p50": int(p50),
        "p90": int(p90),
        "p95": int(p95),
        "p99": int(p99),
        "mean": float(lens.mean()),
        "max": int(lens.max()),
        "pct_over_max_len": float((lens > max_len).mean()),
        "num_over_max_len": int((lens > max_len).sum()),
        "max_len": int(max_len),

    }
    return stats

if __name__ == "__main__":
    from transformers import AutoTokenizer

    train_ds, test_ds, label_names = load_banking77()
    print("train size:", len(train_ds))
    print("test size:", len(test_ds))
    print("num labels:", len(label_names))

    stats = get_label_stats(train_ds)
    print("label stats:", stats)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    len_stats = get_token_length_stats(train_ds, tokenizer)
    print("token length stats:", len_stats)
