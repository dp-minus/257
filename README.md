## Roadblocks
- With the current implementation there might be some data leakage from the `end_day` observation windows.
- Need to clarify/verify the evaluation setup before drawing conclusions.
- Anomaly scores haven't improved baseline performance so far (Not sure yet, but maybe the observation windows are affecting the results.).

# Anomaly-Aware Difficulty Modeling for Player Churn and Retention
Can anomaly detection find unusual player behavior and difficulty patterns that
aren’t normally captured by standard churn-prediction features?

## Data and references

- https://cloud.tsinghua.edu.cn/d/3a13277acd6b4daab1f3/
- https://github.com/JiayuLi-997/DAF-for-churn
- https://jiayuli-997.github.io/files/KDD2021-LiJY-Churn_Prediction.pdf
- https://github.com/Minqi824/ADBench

Data isn't included, please download the following and add to `data/`:
- `interactions.csv` - Tsinghua.
- `labels.csv`, `dataset_split/fold-1` to `fold-5` (`train.uid.npy`, `dev.uid.npy`) from `data/`.

## Install + Run

```bash
pip install -r requirements.txt
python src/experiment.py
```
Outputs write to `results/`.