import argparse
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from features import load_dataset

FOLDS = range(1, 6)

def prep(X_train, X_test):
    imp = SimpleImputer(strategy="median").fit(X_train)
    sc = StandardScaler().fit(imp.transform(X_train))
    return sc.transform(imp.transform(X_train)), sc.transform(imp.transform(X_test))

def anomaly_scores(Xtr, Xte, seed):
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr)
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Xtr)
    return {
        "iforest": (-iso.score_samples(Xtr), -iso.score_samples(Xte)),
        "lof": (-lof.score_samples(Xtr), -lof.score_samples(Xte)),
    }

def eval(Xtr, ytr, Xte, yte, seed):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(yte, prob),
        "pr_auc": average_precision_score(yte, prob),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred),
        "f1": f1_score(yte, pred),
    }

def run_f(X, y, fold, data_dir):
    folder = f"{data_dir}/dataset_split/fold-{fold}"
    train_ids = np.load(f"{folder}/train.uid.npy")
    test_ids = np.load(f"{folder}/dev.uid.npy")
    Xtr_raw, ytr = X.loc[train_ids], y.loc[train_ids]
    Xte_raw, yte = X.loc[test_ids], y.loc[test_ids]
    Xtr, Xte = prep(Xtr_raw, Xte_raw)
    scores = anomaly_scores(Xtr, Xte, fold)
    rows = [{"fold": fold, "model": "baseline", **eval(Xtr, ytr, Xte, yte, fold)}]
    for name, (s_tr, s_te) in scores.items():
        Xtr_e = np.column_stack([Xtr, s_tr])
        Xte_e = np.column_stack([Xte, s_te])
        rows.append({"fold": fold, "model": f"baseline+{name}", **eval(Xtr_e, ytr, Xte_e, yte, fold)})
    anomaly_auc = {n: roc_auc_score(yte, s[1]) for n, s in scores.items()}
    return rows, (Xtr_raw, ytr, yte, scores), anomaly_auc

def plot_dist(s_tr, s_te, out):
    plt.figure(figsize=(6, 4))
    plt.hist(s_tr, bins=40, alpha=0.6, label="train", density=True)
    plt.hist(s_te, bins=40, alpha=0.6, label="test", density=True)
    plt.xlabel("Isolation Forest anomaly score")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_vs_churn(score, y, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].boxplot([score[y == 0], score[y == 1]], tick_labels=["retained", "churned"])
    ax[0].set_ylabel("anomaly score")
    decile = pd.qcut(pd.Series(score).rank(method="first"), 10, labels=False)
    rate = pd.Series(y).groupby(decile.values).mean()
    ax[1].bar(rate.index + 1, rate.values)
    ax[1].set_xlabel("anomaly score decile (10 = most anomalous)")
    ax[1].set_ylabel("churn rate")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return rate

def plot_compare(summary, out):
    metrics = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    mean = summary.xs("mean", axis=1, level=1)[metrics]
    std = summary.xs("std", axis=1, level=1)[metrics]
    ax = mean.T.plot.bar(yerr=std.T, figsize=(8, 4), rot=0, capsize=2)
    ax.set_ylabel("score (mean over splits)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def feature_differences(X, score, top=0.1):
    flagged = score >= np.quantile(score, 1 - top)
    Xs = (X - X.mean()) / X.std()
    diff = Xs[flagged].mean() - Xs[~flagged].mean()
    return diff.sort_values(key=abs, ascending=False)

def write_analysis(out_dir, n, churn, summary, a_auc, rate, diff, s_te, yte):
    lines = [f"Players: {n}, churn rate: {churn:.3f}", ""]
    lines.append("Anomaly score alone vs churn (held-out ROC-AUC, mean over folds):")
    for name, v in a_auc.items():
        lines.append(f"  {name}: {v:.3f}")
    lines.append(f"Mean iforest score (fold 1 held-out), churned: {s_te[yte == 1].mean():.3f}, "f"retained: {s_te[yte == 0].mean():.3f}")
    lines.append("Churn rate by iforest score decile (fold 1 held-out): "+ ", ".join(f"{r:.2f}" for r in rate.values))
    lines += ["", "Standardized feature difference, top 10% anomalous vs rest (fold 1 train):"]
    lines += [f"  {k}: {v:+.2f}" for k, v in diff.head(6).items()]
    lines += ["", "Churn prediction (mean/std over 5 folds):", summary.round(3).to_string()]
    text = "\n".join(lines)
    with open(f"{out_dir}/analysis.txt", "w") as f:
        f.write(text)
    print(text)

def main(data_dir, out_dir):
    X, y = load_dataset(data_dir)
    print(f"{len(X)} players, churn rate {y.mean():.3f}")
    all_rows, anomaly_aucs = [], []
    for fold in FOLDS:
        rows, parts, a_auc = run_f(X, y, fold, data_dir)
        all_rows += rows
        anomaly_aucs.append(a_auc)
        if fold == 1:
            first = parts
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(f"{out_dir}/metrics_per_fold.csv", index=False)
    summary = metrics.drop(columns="fold").groupby("model").agg(["mean", "std"])
    summary.to_csv(f"{out_dir}/metrics_summary.csv")
    plot_compare(summary, f"{out_dir}/model_comparison.png")
    Xtr_raw, ytr, yte, scores = first
    s_tr, s_te = scores["iforest"]
    plot_dist(s_tr, s_te, f"{out_dir}/anomaly_distribution.png")
    rate = plot_vs_churn(s_te, yte.values, f"{out_dir}/anomaly_vs_churn.png")
    diff = feature_differences(Xtr_raw.reset_index(drop=True), s_tr)
    diff.to_csv(f"{out_dir}/anomalous_feature_differences.csv", header=["std_diff"])
    a_auc = pd.DataFrame(anomaly_aucs).mean()
    write_analysis(out_dir, len(X), y.mean(), summary, a_auc, rate, diff, s_te, yte.values)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
