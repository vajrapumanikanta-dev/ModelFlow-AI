"""
Module 13: Experiment Tracking (core/tracker.py)
Stores and compares model training runs in a local SQLite database.
"""

import os
import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


class ExperimentTracker:
    """
    Manages experiment metadata, training parameters, evaluation metrics,
    and side-by-side run comparisons.
    """

    def __init__(self, db_path: str = "experiments/experiments.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Creates the experiments table if not already present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT,
                    timestamp TEXT,
                    dataset_name TEXT,
                    problem_type TEXT,
                    target_col TEXT,
                    model_name TEXT,
                    features TEXT,
                    hyperparameters TEXT,
                    train_duration_sec REAL,
                    cv_score_mean REAL,
                    cv_score_std REAL,
                    test_metrics TEXT,
                    model_path TEXT,
                    notes TEXT
                )
                """
            )
            conn.commit()

    def log_run(
        self,
        model_name: str,
        problem_type: str,
        dataset_name: str,
        features: List[str],
        test_metrics: Dict[str, Any],
        target_col: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        cv_score: Optional[Dict[str, Any]] = None,
        train_duration_sec: float = 0.0,
        model_path: Optional[str] = None,
        experiment_name: str = "Default_Experiment",
        notes: str = "",
    ) -> str:
        """
        Logs a single experiment run to the database and returns the generated run_id.
        """
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        features_json = json.dumps(features)
        params_json = json.dumps(hyperparameters or {})
        metrics_json = json.dumps(test_metrics or {})

        cv_mean = float(cv_score.get("mean", 0.0)) if cv_score else 0.0
        cv_std = float(cv_score.get("std", 0.0)) if cv_score else 0.0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (
                    run_id, experiment_name, timestamp, dataset_name,
                    problem_type, target_col, model_name, features,
                    hyperparameters, train_duration_sec, cv_score_mean,
                    cv_score_std, test_metrics, model_path, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_name,
                    timestamp,
                    dataset_name,
                    problem_type,
                    target_col or "",
                    model_name,
                    features_json,
                    params_json,
                    train_duration_sec,
                    cv_mean,
                    cv_std,
                    metrics_json,
                    model_path or "",
                    notes,
                ),
            )
            conn.commit()

        return run_id

    def list_runs(self, experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Returns all logged runs, optionally filtered by experiment name.
        """
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if experiment_name:
                cursor.execute("SELECT * FROM runs WHERE experiment_name = ? ORDER BY timestamp DESC", (experiment_name,))
            else:
                cursor.execute("SELECT * FROM runs ORDER BY timestamp DESC")
            rows = cursor.fetchall()

        runs = []
        for r in rows:
            d = dict(r)
            d["features"] = json.loads(d["features"]) if d["features"] else []
            d["hyperparameters"] = json.loads(d["hyperparameters"]) if d["hyperparameters"] else {}
            d["test_metrics"] = json.loads(d["test_metrics"]) if d["test_metrics"] else {}
            runs.append(d)
        return runs

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single run by run_id."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d["features"] = json.loads(d["features"]) if d["features"] else []
            d["hyperparameters"] = json.loads(d["hyperparameters"]) if d["hyperparameters"] else {}
            d["test_metrics"] = json.loads(d["test_metrics"]) if d["test_metrics"] else {}
            return d

    def delete_run(self, run_id: str) -> bool:
        """Deletes a run record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Constructs a side-by-side comparison table of selected runs.
        """
        if not run_ids:
            return pd.DataFrame()

        rows = []
        for rid in run_ids:
            run = self.get_run(rid)
            if not run:
                continue

            row_data = {
                "Run ID": run["run_id"],
                "Model": run["model_name"],
                "Problem Type": run["problem_type"],
                "Dataset": run["dataset_name"],
                "Timestamp": run["timestamp"],
                "CV Score": f"{run['cv_score_mean']:.4f} ± {run['cv_score_std']:.4f}",
                "Train Time (s)": round(run["train_duration_sec"], 3),
                "Num Features": len(run["features"]),
            }

            # Flatten test metrics
            for k, v in run["test_metrics"].items():
                if isinstance(v, (int, float)):
                    row_data[k.upper()] = round(v, 4)

            rows.append(row_data)

        return pd.DataFrame(rows)
