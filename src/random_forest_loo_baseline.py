from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET_CONFIGS = {
    "acl_rsi": {
        "display_name": "ACL-RSI",
        "json_file": "acl_rsi_predict_report.json",
        "ground_truth_key": "ground_truth_ACL-RSI",
        "threshold": 65.0,
        "prediction_min": 0.0,
        "prediction_max": 100.0,
        "prediction_output": "rf_acl_rsi_loo_predictions.csv",
        "metrics_output": "rf_acl_rsi_loo_metrics.json",
    },
    "tegner": {
        "display_name": "Tegner",
        "json_file": "tegner_predict_report.json",
        "ground_truth_key": "ground_truth_Tegner",
        "threshold": 6.0,
        "prediction_min": 0.0,
        "prediction_max": 10.0,
        "prediction_output": "rf_tegner_loo_predictions.csv",
        "metrics_output": "rf_tegner_loo_metrics.json",
    },
}


NARRATIVE_FIELDS = (
    ("chief_complaint", "Chief complaint"),
    ("history_of_present_illness", "History of present illness"),
    ("past_medical_history", "Past medical history"),
    ("personal_history", "Personal history"),
)

MISSING_TEXT_VALUES = {"", "nan", "none", "null", "na", "n/a", "not provided", "empty", "not available"}


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_patient_groups(json_path: Path) -> Dict[str, List[str]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("The cohort file must map center names to patient identifier lists.")
    if any(not isinstance(patients, list) for patients in payload.values()):
        raise ValueError("Each center must contain a list of patient identifiers.")
    return {str(center): [str(patient) for patient in patients] for center, patients in payload.items()}


def build_patient_to_group_map(patient_groups: Dict[str, List[str]]) -> Dict[str, str]:
    patient_to_group: Dict[str, str] = {}
    for group_name, patients in patient_groups.items():
        for patient in patients:
            if patient in patient_to_group:
                raise ValueError("Patient identifiers in the cohort file must be unique.")
            patient_to_group[patient] = group_name
    if len(patient_to_group) < 2:
        raise ValueError("Leave-one-out cross-validation requires at least two patients.")
    return patient_to_group


def canonicalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in MISSING_TEXT_VALUES:
        return None
    replacements = {
        "（": "(",
        "）": ")",
        "／": "/",
        "，": ",",
        "；": ";",
        "：": ":",
        "＋": "+",
        "－": "-",
        "—": "-",
        "～": "~",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_category_value(value: object) -> Optional[str]:
    text = canonicalize_text(value)
    if text is None:
        return None
    return text.lower()


def rankdata_average(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def parse_numeric(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in MISSING_TEXT_VALUES:
        return None

    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if not match:
        return None
    return float(match.group())


def read_admission_record(file_path: Path) -> pd.Series:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            frame = pd.read_csv(file_path, encoding=encoding, dtype=str)
        except (UnicodeError, pd.errors.ParserError):
            continue
        if frame.empty:
            raise ValueError("An admission record contains no patient rows.")
        return frame.iloc[0]
    raise ValueError("An admission record could not be decoded or parsed.")


def safe_row_text(row: pd.Series, column: str) -> str:
    value = row.get(column)
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in MISSING_TEXT_VALUES else text


def load_ground_truth(interim_root: Path, center: str, patient: str, target_key: str) -> float:
    config = TARGET_CONFIGS[target_key]
    file_path = interim_root / center / patient / str(config["json_file"])
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    value = parse_numeric(payload.get(config["ground_truth_key"]))
    if value is None or not np.isfinite(value):
        raise ValueError(f"A {config['display_name']} outcome is missing or non-finite.")
    if not float(config["prediction_min"]) <= value <= float(config["prediction_max"]):
        raise ValueError(f"A {config['display_name']} outcome is outside the score range.")
    return value


def build_feature_table(
    raw_processed_root: Path,
    interim_root: Path,
    patient_to_group: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    seen: set[str] = set()
    for patient_dir in sorted(raw_processed_root.glob("*/*")):
        if not patient_dir.is_dir() or patient_dir.name not in patient_to_group:
            continue
        patient = patient_dir.name
        center = patient_dir.parent.name
        if patient in seen:
            raise ValueError("A cohort patient matches multiple record directories.")
        seen.add(patient)
        admission = read_admission_record(patient_dir / "Admission note.csv")
        sex = normalize_category_value(safe_row_text(admission, "Sex"))
        age = parse_numeric(safe_row_text(admission, "Age"))
        if age is not None and not np.isfinite(age):
            age = None
        narratives = {name: safe_row_text(admission, label) for name, label in NARRATIVE_FIELDS}
        narrative_text = "\n".join(
            f"{label}: {narratives[name]}" for name, label in NARRATIVE_FIELDS if narratives[name]
        )
        available_count = int(sex is not None) + int(age is not None) + sum(bool(v) for v in narratives.values())
        if available_count == 0:
            raise ValueError("A cohort patient has no usable admission-stage predictors.")
        rows.append({
            "center": center,
            "referral_center": patient_to_group[patient],
            "patient": patient,
            "sex": sex,
            "age": age,
            **narratives,
            "model_input_text": narrative_text,
            "input_char_count": len(narrative_text),
            "available_feature_count": available_count,
            "acl_rsi_ground_truth": load_ground_truth(interim_root, center, patient, "acl_rsi"),
            "tegner_ground_truth": load_ground_truth(interim_root, center, patient, "tegner"),
        })
    if seen != set(patient_to_group):
        raise ValueError(f"Admission records are missing for {len(set(patient_to_group) - seen)} cohort patients.")
    return pd.DataFrame(rows)


def normalize_model_input_text(text: object) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def iter_character_ngrams(text: object, min_n: int, max_n: int) -> List[str]:
    normalized = normalize_model_input_text(text).replace(" ", "")
    if not normalized:
        return []

    ngrams: List[str] = []
    for n in range(max(1, min_n), max(min_n, max_n) + 1):
        if len(normalized) < n:
            continue
        for index in range(len(normalized) - n + 1):
            ngrams.append(normalized[index : index + n])

    if not ngrams:
        return [normalized]
    return ngrams


def hash_token_to_index(token: str, n_features: int) -> int:
    return zlib.crc32(token.encode("utf-8")) % n_features


def encode_text_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
    n_features: int,
    min_n: int,
    max_n: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    def encode_rows(texts: Sequence[object]) -> np.ndarray:
        matrix = np.zeros((len(texts), n_features), dtype=float)
        for row_index, text in enumerate(texts):
            ngrams = iter_character_ngrams(text, min_n=min_n, max_n=max_n)
            if not ngrams:
                continue

            counts: Dict[int, int] = {}
            for ngram in ngrams:
                feature_index = hash_token_to_index(ngram, n_features=n_features)
                counts[feature_index] = counts.get(feature_index, 0) + 1

            total = float(len(ngrams))
            for feature_index, count in counts.items():
                matrix[row_index, feature_index] = count / total
        return matrix

    encoded_names = [f"text_hash_{index:04d}" for index in range(n_features)]
    train_matrix = encode_rows(train_df[text_column].fillna("").tolist())
    test_matrix = encode_rows(test_df[text_column].fillna("").tolist())
    return train_matrix, test_matrix, encoded_names


def encode_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_vectorizer_params: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    train_age = pd.to_numeric(train_df["age"], errors="coerce").to_numpy(dtype=float)
    test_age = pd.to_numeric(test_df["age"], errors="coerce").to_numpy(dtype=float)
    finite_train_age = train_age[np.isfinite(train_age)]
    median_age = float(np.median(finite_train_age)) if finite_train_age.size else 0.0
    train_columns = [np.where(np.isfinite(train_age), train_age, median_age), (~np.isfinite(train_age)).astype(float)]
    test_columns = [np.where(np.isfinite(test_age), test_age, median_age), (~np.isfinite(test_age)).astype(float)]
    names = ["age", "age__missing"]

    train_sex = [normalize_category_value(value) for value in train_df["sex"]]
    test_sex = [normalize_category_value(value) for value in test_df["sex"]]
    categories = sorted({value for value in train_sex if value is not None})
    for category in categories:
        train_columns.append(np.array([value == category for value in train_sex], dtype=float))
        test_columns.append(np.array([value == category for value in test_sex], dtype=float))
        names.append(f"sex__{category}")
    train_columns.extend([
        np.array([value is None for value in train_sex], dtype=float),
        np.zeros(len(train_df), dtype=float),
    ])
    test_columns.extend([
        np.array([value is None for value in test_sex], dtype=float),
        np.array([value is not None and value not in categories for value in test_sex], dtype=float),
    ])
    names.extend(["sex__missing", "sex__other"])

    train_text, test_text, text_names = encode_text_matrices(
        train_df,
        test_df,
        text_column="model_input_text",
        n_features=text_vectorizer_params["hash_dim"],
        min_n=text_vectorizer_params["ngram_min"],
        max_n=text_vectorizer_params["ngram_max"],
    )
    return (
        np.column_stack([*train_columns, train_text]),
        np.column_stack([*test_columns, test_text]),
        names + text_names,
    )


class DecisionTreeRegressorScratch:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.n_features_: Optional[int] = None
        self.tree_: Optional[Dict[str, object]] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressorScratch":
        self.n_features_ = x.shape[1]
        self.tree_ = self._build_tree(x, y, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("The tree has not been fitted.")
        return np.array([self._predict_row(row, self.tree_) for row in x], dtype=float)

    def _predict_row(self, row: np.ndarray, node: Dict[str, object]) -> float:
        if node["type"] == "leaf":
            return float(node["value"])
        if row[int(node["feature"])] <= float(node["threshold"]):
            return self._predict_row(row, node["left"])
        return self._predict_row(row, node["right"])

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> Dict[str, object]:
        prediction = float(np.mean(y))
        if len(y) < self.min_samples_split:
            return {"type": "leaf", "value": prediction}
        if self.max_depth is not None and depth >= self.max_depth:
            return {"type": "leaf", "value": prediction}
        if np.allclose(y, y[0]):
            return {"type": "leaf", "value": prediction}

        feature_indices = np.arange(self.n_features_)
        if self.max_features is not None and self.max_features < len(feature_indices):
            feature_indices = self.rng.choice(feature_indices, size=self.max_features, replace=False)

        best = self._best_split(x, y, feature_indices)
        if best is None:
            return {"type": "leaf", "value": prediction}

        feature_index, threshold = best
        left_mask = x[:, feature_index] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return {"type": "leaf", "value": prediction}

        return {
            "type": "node",
            "value": prediction,
            "feature": int(feature_index),
            "threshold": float(threshold),
            "left": self._build_tree(x[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(x[right_mask], y[right_mask], depth + 1),
        }

    def _best_split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
    ) -> Optional[Tuple[int, float]]:
        best_feature = None
        best_threshold = None
        best_loss = math.inf

        for feature_index in feature_indices:
            values = x[:, feature_index]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_y = y[order]

            unique_count = np.unique(sorted_values).size
            if unique_count <= 1:
                continue

            cumulative_sum = np.cumsum(sorted_y)
            cumulative_sq_sum = np.cumsum(sorted_y**2)
            total_sum = cumulative_sum[-1]
            total_sq_sum = cumulative_sq_sum[-1]

            for split_idx in range(self.min_samples_leaf, len(sorted_y) - self.min_samples_leaf + 1):
                if split_idx >= len(sorted_y):
                    break
                if sorted_values[split_idx - 1] == sorted_values[split_idx]:
                    continue

                left_count = split_idx
                right_count = len(sorted_y) - split_idx

                left_sum = cumulative_sum[split_idx - 1]
                left_sq_sum = cumulative_sq_sum[split_idx - 1]
                right_sum = total_sum - left_sum
                right_sq_sum = total_sq_sum - left_sq_sum

                left_sse = left_sq_sum - (left_sum**2) / left_count
                right_sse = right_sq_sum - (right_sum**2) / right_count
                loss = left_sse + right_sse

                if loss < best_loss:
                    best_loss = float(loss)
                    best_feature = int(feature_index)
                    best_threshold = float((sorted_values[split_idx - 1] + sorted_values[split_idx]) / 2.0)

        if best_feature is None or best_threshold is None:
            return None
        return best_feature, best_threshold


class RandomForestRegressorScratch:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 6,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        max_features: str | int | None = "sqrt",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: List[DecisionTreeRegressorScratch] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestRegressorScratch":
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = x.shape
        max_features = self._resolve_max_features(n_features)

        self.trees_ = []
        for _ in range(self.n_estimators):
            bootstrap_idx = rng.integers(0, n_samples, size=n_samples)
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=int(rng.integers(0, 10**9)),
            )
            tree.fit(x[bootstrap_idx], y[bootstrap_idx])
            self.trees_.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("The forest has not been fitted.")
        predictions = np.vstack([tree.predict(x) for tree in self.trees_])
        return np.mean(predictions, axis=0)

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return max(1, min(n_features, self.max_features))
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(math.log2(max(n_features, 2))))
        return n_features


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return pearson_correlation(rankdata_average(y_true), rankdata_average(y_pred))


def roc_auc_score_manual(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    positive_mask = y_true_binary == 1
    negative_mask = y_true_binary == 0
    n_pos = int(np.sum(positive_mask))
    n_neg = int(np.sum(negative_mask))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_average(y_score)
    positive_ranks = ranks[positive_mask]
    auc = (np.sum(positive_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    auc = roc_auc_score_manual(y_true_binary, y_pred)

    return {
        "AUC": auc,
        "Recall": float(recall),
        "Precision": float(precision),
        "F1-Score": float(f1),
        "Positive Count (GT)": int(np.sum(y_true_binary)),
        "Negative Count (GT)": int(np.sum(1 - y_true_binary)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    values: List[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        metric_value = metric_fn(y_true[sample_idx], y_pred[sample_idx])
        if metric_value is None or not np.isfinite(metric_value):
            continue
        values.append(float(metric_value))

    if not values:
        return {
            "lower": None,
            "upper": None,
            "valid_bootstraps": 0,
            "total_bootstraps": n_bootstrap,
        }

    return {
        "lower": float(np.percentile(values, 2.5)),
        "upper": float(np.percentile(values, 97.5)),
        "valid_bootstraps": len(values),
        "total_bootstraps": n_bootstrap,
    }


def bootstrap_ci_stratified_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_binary = (y_true >= threshold).astype(int)

    positive_idx = np.where(y_true_binary == 1)[0]
    negative_idx = np.where(y_true_binary == 0)[0]

    values: List[float] = []
    for _ in range(n_bootstrap):
        sampled_positive = rng.choice(positive_idx, size=len(positive_idx), replace=True)
        sampled_negative = rng.choice(negative_idx, size=len(negative_idx), replace=True)
        sampled_idx = np.concatenate([sampled_positive, sampled_negative])
        metric_value = metric_fn(y_true[sampled_idx], y_pred[sampled_idx])
        if metric_value is None or not np.isfinite(metric_value):
            continue
        values.append(float(metric_value))

    if not values:
        return {
            "lower": None,
            "upper": None,
            "valid_bootstraps": 0,
            "total_bootstraps": n_bootstrap,
        }

    return {
        "lower": float(np.percentile(values, 2.5)),
        "upper": float(np.percentile(values, 97.5)),
        "valid_bootstraps": len(values),
        "total_bootstraps": n_bootstrap,
    }


def metric_bundle(value: float, ci: Dict[str, object]) -> Dict[str, object]:
    return {
        "value": None if value is None or not np.isfinite(value) else float(value),
        "95% CI": ci,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int, seed: int) -> Dict[str, Dict[str, object]]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    gt_range = float(np.max(y_true) - np.min(y_true))
    nrmse = float(rmse / gt_range) if gt_range > 0 else float("nan")
    pearson = pearson_correlation(y_true, y_pred)
    spearman = spearman_correlation(y_true, y_pred)

    return {
        "MAE": metric_bundle(
            mae,
            bootstrap_ci(y_true, y_pred, lambda a, b: float(np.mean(np.abs(a - b))), n_bootstrap, seed + 11),
        ),
        "RMSE": metric_bundle(
            rmse,
            bootstrap_ci(y_true, y_pred, lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))), n_bootstrap, seed + 23),
        ),
        "NRMSE (by range)": metric_bundle(
            nrmse,
            bootstrap_ci(
                y_true,
                y_pred,
                lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / (np.max(a) - np.min(a))) if np.max(a) > np.min(a) else float("nan"),
                n_bootstrap,
                seed + 37,
            ),
        ),
        "Pearson Correlation": metric_bundle(
            pearson,
            bootstrap_ci(y_true, y_pred, pearson_correlation, n_bootstrap, seed + 41),
        ),
        "Spearman Correlation": metric_bundle(
            spearman,
            bootstrap_ci(y_true, y_pred, spearman_correlation, n_bootstrap, seed + 53),
        ),
    }


def classification_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    point_metrics = classification_metrics(y_true, y_pred, threshold)
    y_true_binary = (y_true >= threshold).astype(int)

    y_pred_binary = (y_pred >= threshold).astype(int)
    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))

    return {
        "Confusion Matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "Threshold": threshold,
        "AUC": metric_bundle(
            point_metrics["AUC"],
            bootstrap_ci_stratified_binary(
                y_true,
                y_pred,
                threshold,
                lambda a, b: roc_auc_score_manual((a >= threshold).astype(int), b),
                n_bootstrap,
                seed + 61,
            ),
        ),
        "Recall": metric_bundle(
            point_metrics["Recall"],
            bootstrap_ci_stratified_binary(
                y_true,
                y_pred,
                threshold,
                lambda a, b: classification_metrics(a, b, threshold)["Recall"],
                n_bootstrap,
                seed + 71,
            ),
        ),
        "Precision": metric_bundle(
            point_metrics["Precision"],
            bootstrap_ci_stratified_binary(
                y_true,
                y_pred,
                threshold,
                lambda a, b: classification_metrics(a, b, threshold)["Precision"],
                n_bootstrap,
                seed + 83,
            ),
        ),
        "F1-Score": metric_bundle(
            point_metrics["F1-Score"],
            bootstrap_ci_stratified_binary(
                y_true,
                y_pred,
                threshold,
                lambda a, b: classification_metrics(a, b, threshold)["F1-Score"],
                n_bootstrap,
                seed + 97,
            ),
        ),
        "Positive Count (GT)": int(np.sum(y_true_binary)),
        "Negative Count (GT)": int(np.sum(1 - y_true_binary)),
    }


def save_json(output_path: Path, payload: Dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def loo_predict(
    df: pd.DataFrame,
    target_column: str,
    model_params: Dict[str, object],
    text_vectorizer_params: Dict[str, int],
    seed: int,
) -> Tuple[pd.DataFrame, List[str]]:
    if len(df) < 2:
        raise ValueError("Leave-one-out cross-validation requires at least two patients.")
    records: List[Dict[str, object]] = []
    observed_encoded_names: set[str] = set()
    for test_index in range(len(df)):
        train_mask = np.arange(len(df)) != test_index
        train_df = df.iloc[train_mask].reset_index(drop=True)
        test_df = df.iloc[[test_index]].reset_index(drop=True)
        x_train, x_test, encoded_names = encode_feature_matrices(train_df, test_df, text_vectorizer_params)
        observed_encoded_names.update(encoded_names)
        model = RandomForestRegressorScratch(**model_params, random_state=seed + test_index)
        model.fit(x_train, train_df[target_column].to_numpy(dtype=float))
        prediction = float(model.predict(x_test)[0])
        row = test_df.iloc[0]
        ground_truth = float(row[target_column])
        records.append({
            "center": row["center"],
            "referral_center": row["referral_center"],
            "patient": row["patient"],
            "ground_truth": ground_truth,
            "raw_prediction": prediction,
            "prediction": prediction,
            "residual": ground_truth - prediction,
            "absolute_error": abs(ground_truth - prediction),
            "training_patient_count": len(train_df),
            "available_feature_count": int(row["available_feature_count"]),
        })
    return pd.DataFrame(records), sorted(observed_encoded_names)


def build_metric_sections(
    predictions_df: pd.DataFrame,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    y_true = predictions_df["ground_truth"].to_numpy(dtype=float)
    y_pred = predictions_df["prediction"].to_numpy(dtype=float)
    return {
        "n_patients": int(len(predictions_df)),
        "regression_metrics": regression_metrics(y_true, y_pred, n_bootstrap=n_bootstrap, seed=seed),
        "classification_metrics": classification_metrics_with_ci(
            y_true,
            y_pred,
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
    }


def evaluate_target(
    feature_table: pd.DataFrame,
    target_key: str,
    patient_groups: Dict[str, List[str]],
    output_dir: Path,
    model_params: Dict[str, object],
    text_vectorizer_params: Dict[str, int],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    config = TARGET_CONFIGS[target_key]
    predictions, encoded_names = loo_predict(
        feature_table,
        target_column=f"{target_key}_ground_truth",
        model_params=model_params,
        text_vectorizer_params=text_vectorizer_params,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / str(config["prediction_output"])
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    partition_metrics: Dict[str, object] = {}
    for offset, group_name in enumerate(patient_groups, start=1):
        group_predictions = predictions[predictions["referral_center"] == group_name]
        if not group_predictions.empty:
            partition_metrics[group_name] = build_metric_sections(
                group_predictions, float(config["threshold"]), n_bootstrap, seed + offset * 100
            )
    metrics_path = output_dir / str(config["metrics_output"])
    save_json(metrics_path, {
        "task": config["display_name"],
        "model": {"name": "RandomForestRegressorScratch", **model_params},
        "evaluation": "Leave-One-Out Cross-Validation (LOO-CV)",
        "n_patients": len(feature_table),
        "training_patients_per_fold": len(feature_table) - 1,
        "seed": seed,
        "bootstrap_repeats": n_bootstrap,
        "feature_columns": [
            {"name": "sex", "label": "Sex", "kind": "categorical"},
            {"name": "age", "label": "Age", "kind": "numeric"},
            *[{"name": name, "label": label, "kind": "text"} for name, label in NARRATIVE_FIELDS],
        ],
        "encoding_summary": {
            "categorical_encoding": "one_hot",
            "numeric_imputation": "training_fold_median",
            "text_vectorizer": text_vectorizer_params,
            "input_char_limit": None,
            "encoded_feature_names_observed": encoded_names,
        },
        "overall_metrics": build_metric_sections(predictions, float(config["threshold"]), n_bootstrap, seed),
        "partition_metrics": partition_metrics,
        "prediction_file": str(prediction_path),
    })
    return {
        "task": config["display_name"],
        "n_patients": len(feature_table),
        "prediction_file": str(prediction_path),
        "metrics_file": str(metrics_path),
    }


def parse_args() -> argparse.Namespace:
    project_root = resolve_project_root()
    parser = argparse.ArgumentParser(description="Random-forest LOO-CV baselines for ACL-RSI and Tegner.")
    parser.add_argument("--raw-processed-root", type=Path, default=project_root / "data" / "raw_processed")
    parser.add_argument("--interim-root", type=Path, default=project_root / "data" / "interim")
    parser.add_argument("--referral-centers-json", type=Path, default=project_root / "data" / "reports" / "patient_referral_centers.json")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data" / "baselines_ml")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-split", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", choices=("sqrt", "log2", "all"), default="sqrt")
    parser.add_argument("--text-hash-dim", type=int, default=256)
    parser.add_argument("--text-ngram-min", type=int, default=2)
    parser.add_argument("--text-ngram-max", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("n_estimators", "max_depth", "min_samples_leaf", "text_hash_dim", "text_ngram_min", "bootstrap_repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.min_samples_split < 2:
        parser.error("--min-samples-split must be at least 2.")
    if args.text_ngram_max < args.text_ngram_min:
        parser.error("--text-ngram-max must be at least --text-ngram-min.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    patient_groups = load_patient_groups(args.referral_centers_json)
    patient_to_group = build_patient_to_group_map(patient_groups)
    feature_table = build_feature_table(args.raw_processed_root, args.interim_root, patient_to_group)
    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": None if args.max_features == "all" else args.max_features,
    }
    text_vectorizer_params = {
        "hash_dim": args.text_hash_dim,
        "ngram_min": args.text_ngram_min,
        "ngram_max": args.text_ngram_max,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_matrix_path = args.output_dir / "rf_feature_matrix.csv"
    feature_table.to_csv(feature_matrix_path, index=False, encoding="utf-8-sig")
    outputs = []
    for offset, target_key in enumerate(("acl_rsi", "tegner")):
        outputs.append(evaluate_target(
            feature_table=feature_table,
            target_key=target_key,
            patient_groups=patient_groups,
            output_dir=args.output_dir,
            model_params=model_params,
            text_vectorizer_params=text_vectorizer_params,
            n_bootstrap=args.bootstrap_repeats,
            seed=args.seed + offset * 1000,
        ))
    save_json(args.output_dir / "rf_loo_run_summary.json", {
        "feature_matrix": str(feature_matrix_path),
        "referral_centers_json": str(args.referral_centers_json),
        "n_patients": len(feature_table),
        "training_patients_per_fold": len(feature_table) - 1,
        "partition_sizes": {center: len(patients) for center, patients in patient_groups.items()},
        "model": model_params,
        "text_vectorizer": text_vectorizer_params,
        "seed": args.seed,
        "bootstrap_repeats": args.bootstrap_repeats,
        "outputs": outputs,
    })
    print("Feature matrix saved to:", feature_matrix_path)
    for output in outputs:
        print(f"{output['task']} predictions saved to:", output["prediction_file"])
        print(f"{output['task']} metrics saved to:", output["metrics_file"])


if __name__ == "__main__":
    main()
