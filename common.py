"""Framework-independent parts shared by medi_parse_tf.py and medi_parse_torch.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent


def preprocess_text_with_line_numbers(filename):
    """Notebook's parser: one dict per sentence with target, lowercased text, line_number, total_lines."""
    abstract_lines, samples = "", []
    for line in open(filename):
        if line.startswith("###"):
            abstract_lines = ""
        elif line.isspace():
            split = abstract_lines.splitlines()
            for n, l in enumerate(split):
                target, text = l.split("\t")
                samples.append({"target": target, "text": text.lower(),
                                "line_number": n, "total_lines": len(split) - 1})
        else:
            abstract_lines += line
    return samples


def load_data(dataset):
    d = ROOT / "data" / "pubmed-rct" / f"PubMed_{dataset}_RCT_numbers_replaced_with_at_sign"
    dfs = [pd.DataFrame(preprocess_text_with_line_numbers(d / f)) for f in ("train.txt", "dev.txt", "test.txt")]
    le = LabelEncoder().fit(dfs[0]["target"])
    ys = [le.transform(df["target"]) for df in dfs]
    print(f"[data] {dataset}: train={len(dfs[0])} val={len(dfs[1])} test={len(dfs[2])} classes={list(le.classes_)}")
    return dfs, ys, list(le.classes_)


def calculate_results(y_true, y_pred):
    """Same as helper_functions.calculate_results used by the notebook (weighted averages)."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred) * 100, "precision": p, "recall": r, "f1": f}


class Results:
    def __init__(self, framework, dataset, y_val, y_test, class_names):
        self.out = ROOT / "results" / framework / dataset
        self.out.mkdir(parents=True, exist_ok=True)
        self.ckpt = ROOT / "checkpoints" / framework / dataset
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.framework, self.dataset, self.y_val, self.y_test, self.names = framework, dataset, y_val, y_test, class_names

    def done(self, mid):
        if (self.out / f"model_{mid}.json").exists():
            print(f"[model {mid}] already done, skipping (delete {self.out}/model_{mid}.json to rerun)")
            return True
        return False

    def save(self, mid, name, val_pred, test_pred, train_secs):
        yv, yt = self.y_val, self.y_test
        res = {"model": mid, "name": name, "framework": self.framework, "dataset": self.dataset,
               "train_seconds": round(train_secs, 1),
               "val": calculate_results(yv, val_pred), "test": calculate_results(yt, test_pred),
               "val_report": classification_report(yv, val_pred, target_names=self.names, output_dict=True),
               "test_report": classification_report(yt, test_pred, target_names=self.names, output_dict=True),
               "val_confusion": confusion_matrix(yv, val_pred).tolist(),
               "test_confusion": confusion_matrix(yt, test_pred).tolist()}
        (self.out / f"model_{mid}.json").write_text(json.dumps(res, indent=2))
        print(f"[model {mid}] {name}: val acc={res['val']['accuracy']:.2f} f1={res['val']['f1']:.4f} | "
              f"test acc={res['test']['accuracy']:.2f} f1={res['test']['f1']:.4f} ({train_secs/60:.1f} min)", flush=True)


def line_number_one_hot(train_df, *dfs):
    """Model 7's OneHotEncoder on line_number: categories = values seen in train (31 for 20k)."""
    cats = np.sort(train_df["line_number"].unique())
    idx = {c: i for i, c in enumerate(cats)}
    out = []
    for df in (train_df, *dfs):
        m = np.zeros((len(df), len(cats)), np.float32)
        rows = [(r, idx[v]) for r, v in enumerate(df["line_number"]) if v in idx]  # unseen -> all zeros
        if rows:
            r, c = zip(*rows); m[list(r), list(c)] = 1.0
        out.append(m)
    return out
