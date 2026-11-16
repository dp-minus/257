import numpy as np
import pandas as pd

def load_logs(data_dir):
    logs = pd.read_csv(f"{data_dir}/interactions.csv")
    print(logs.dtypes, logs.shape, sep="\n")
    return logs

def clean_logs(logs):
    logs = logs.copy()
    for c in ["win", "duration", "retry_time", "session_depth", "timestamp"]:
        logs[c] = pd.to_numeric(logs[c], errors="coerce")
    logs["time"] = pd.to_datetime(logs["time"], errors="coerce")
    logs = logs.dropna(subset=["user_id", "level_id", "session_id", "win", "timestamp", "time"])
    logs = logs[logs["duration"].isna() | (logs["duration"] >= 0)]
    logs = logs[logs["retry_time"].isna() | (logs["retry_time"] >= 0)]
    logs = logs.sort_values(["user_id", "timestamp"])
    date = (logs["time"] - pd.Timedelta(hours=4)).dt.normalize()
    logs["day"] = (date - date.groupby(logs["user_id"]).transform("min")).dt.days
    return logs

def fail_streaks(logs):
    lost = 1 - logs["win"]
    block = logs["win"].groupby(logs["user_id"]).cumsum()
    run = lost.groupby([logs["user_id"], block]).cumsum()
    return run.groupby(logs["user_id"]).max()

def stuck_levels(logs):
    g = logs.groupby(["user_id", "level_id"])["win"].agg(["size", "max"])
    stuck = (g["size"] >= 3) & (g["max"] == 0)
    return stuck.groupby("user_id").sum()

def build_features(logs, end_day):
    logs = logs[logs["day"] < logs["user_id"].map(end_day)]
    g = logs.groupby("user_id")
    feats = pd.DataFrame({
        "n_plays": g.size(),
        "win_rate": g["win"].mean(),
        "avg_duration": g["duration"].mean(),
        "duration_std": g["duration"].std(),
        "avg_retry": g["retry_time"].mean(),
        "retry_std": g["retry_time"].std(),
        "retry_frac": (logs["retry_time"] > 0).groupby(logs["user_id"]).mean(),
        "n_levels": g["level_id"].nunique(),
        "n_levels_won": logs[logs["win"] == 1].groupby("user_id")["level_id"].nunique(),
        "n_sessions": g["session_id"].nunique(),
        "avg_session_depth": g["session_depth"].mean(),
        "max_fail_streak": fail_streaks(logs),
        "n_stuck_levels": stuck_levels(logs),
    })
    feats["n_levels_won"] = feats["n_levels_won"].fillna(0)
    feats["plays_per_level"] = feats["n_plays"] / feats["n_levels"]
    feats[["duration_std", "retry_std"]] = feats[["duration_std", "retry_std"]].fillna(0)
    return feats.replace([np.inf, -np.inf], np.nan), logs

def load_dataset(data_dir):
    logs = clean_logs(load_logs(data_dir))
    labels = pd.read_csv(f"{data_dir}/labels.csv").set_index("user_id")
    feats, window = build_features(logs, labels["end_day"])
    assert (window["day"] < window["user_id"].map(labels["end_day"])).all()
    data = feats.join(labels["label"], how="inner")
    print(f"{len(data)} players, max day used {window['day'].max()}")
    return data.drop(columns="label"), data["label"].astype(int)