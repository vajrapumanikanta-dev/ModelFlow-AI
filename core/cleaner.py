"""
Module 3: Cleaning (core/cleaner.py)
Interactive, user-controlled data cleaning engine with full history logging,
undo capability, and deterministic log replay.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class DataCleaner:
    """
    Stateful dataset cleaning manager that tracks all transformations
    and ensures the user stays in full control.
    """

    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.current_df = df.copy()
        self.history: List[Tuple[pd.DataFrame, Dict[str, Any]]] = []
        self.log: List[Dict[str, Any]] = []

    def _record_state(self, action_type: str, columns: List[str], params: Dict[str, Any], description: str):
        """Saves current state to history for undo and logs the action."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action_type,
            "columns": columns,
            "params": params,
            "description": description,
            "before_shape": (self.current_df.shape[0], self.current_df.shape[1]),
        }
        self.history.append((self.current_df.copy(), entry))

    def _finalize_entry(self):
        """Updates the shape after operation and adds entry to active log."""
        if self.history:
            _, entry = self.history[-1]
            entry["after_shape"] = (self.current_df.shape[0], self.current_df.shape[1])
            self.log.append(entry)

    def undo(self) -> bool:
        """Reverts the last cleaning operation. Returns True if undone, False if at initial state."""
        if not self.history:
            return False
        prev_df, _ = self.history.pop()
        self.current_df = prev_df
        if self.log:
            self.log.pop()
        return True

    def reset(self):
        """Resets dataframe to the original uncleaned state."""
        self.current_df = self.original_df.copy()
        self.history.clear()
        self.log.clear()

    def get_cleaning_log(self) -> List[Dict[str, Any]]:
        """Returns the chronological cleaning log."""
        return list(self.log)

    def handle_duplicates(self, subset: Optional[List[str]] = None, keep: str = "first") -> int:
        """
        Handles duplicate rows.
        keep: 'first', 'last', or 'none' (drop all duplicates).
        """
        keep_val: Union[str, bool] = keep if keep in ["first", "last"] else False
        num_dups = int(self.current_df.duplicated(subset=subset, keep=keep_val).sum())
        
        if num_dups == 0:
            return 0

        self._record_state(
            action_type="remove_duplicates",
            columns=subset if subset else list(self.current_df.columns),
            params={"subset": subset, "keep": keep},
            description=f"Removed {num_dups} duplicate rows (keep='{keep}').",
        )
        self.current_df = self.current_df.drop_duplicates(subset=subset, keep=keep_val).reset_index(drop=True)
        self._finalize_entry()
        return num_dups

    def handle_missing(
        self,
        columns: List[str],
        strategy: str = "mean",
        fill_value: Any = None,
    ) -> Dict[str, Any]:
        """
        Handles missing values for specified columns.
        strategy: 'drop_rows', 'drop_cols', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'constant'
        """
        if not columns:
            return {"affected_rows": 0, "columns": []}

        valid_cols = [c for c in columns if c in self.current_df.columns]
        if not valid_cols:
            return {"affected_rows": 0, "columns": []}

        desc = f"Handled missing values in {len(valid_cols)} columns using strategy '{strategy}'."
        self._record_state(
            action_type="handle_missing",
            columns=valid_cols,
            params={"strategy": strategy, "fill_value": fill_value},
            description=desc,
        )

        affected = 0
        if strategy == "drop_rows":
            before = len(self.current_df)
            self.current_df = self.current_df.dropna(subset=valid_cols).reset_index(drop=True)
            affected = before - len(self.current_df)
        elif strategy == "drop_cols":
            self.current_df = self.current_df.drop(columns=valid_cols)
        elif strategy == "forward_fill":
            self.current_df[valid_cols] = self.current_df[valid_cols].ffill()
        elif strategy == "backward_fill":
            self.current_df[valid_cols] = self.current_df[valid_cols].bfill()
        elif strategy == "mean":
            for col in valid_cols:
                if pd.api.types.is_numeric_dtype(self.current_df[col]):
                    m = self.current_df[col].mean()
                    self.current_df[col] = self.current_df[col].fillna(m)
        elif strategy == "median":
            for col in valid_cols:
                if pd.api.types.is_numeric_dtype(self.current_df[col]):
                    m = self.current_df[col].median()
                    self.current_df[col] = self.current_df[col].fillna(m)
        elif strategy == "mode":
            for col in valid_cols:
                m_series = self.current_df[col].mode(dropna=True)
                if not m_series.empty:
                    self.current_df[col] = self.current_df[col].fillna(m_series.iloc[0])
        elif strategy == "constant":
            for col in valid_cols:
                val = fill_value
                if val is None:
                    val = "Missing" if not pd.api.types.is_numeric_dtype(self.current_df[col]) else 0
                self.current_df[col] = self.current_df[col].fillna(val)

        self._finalize_entry()
        return {"affected_rows": affected, "columns": valid_cols}

    def detect_outliers(
        self, column: str, method: str = "iqr", threshold: float = 1.5
    ) -> Tuple[pd.Series, Optional[float], Optional[float], int]:
        """
        Detects outliers in a numeric column.
        Returns: (boolean_mask_of_outliers, lower_bound, upper_bound, count).
        """
        if column not in self.current_df.columns or not pd.api.types.is_numeric_dtype(self.current_df[column]):
            return pd.Series(False, index=self.current_df.index), None, None, 0

        s = self.current_df[column].dropna()
        if s.empty:
            return pd.Series(False, index=self.current_df.index), None, None, 0

        if method == "iqr":
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            iqr = q75 - q25
            lower = q25 - threshold * iqr
            upper = q75 + threshold * iqr
        elif method == "zscore":
            mean = float(s.mean())
            std = float(s.std()) if len(s) > 1 else 0.0
            if std == 0:
                return pd.Series(False, index=self.current_df.index), mean, mean, 0
            lower = mean - threshold * std
            upper = mean + threshold * std
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        mask = (self.current_df[column] < lower) | (self.current_df[column] > upper)
        count = int(mask.sum())
        return mask, lower, upper, count

    def handle_outliers(
        self,
        column: str,
        method: str = "iqr",
        threshold: float = 1.5,
        action: str = "cap",
    ) -> int:
        """
        Handles outliers in a numeric column.
        action: 'cap' (winsorize to bounds), 'remove' (drop outlier rows), 'keep' (do nothing).
        """
        if action == "keep":
            return 0

        mask, lower, upper, count = self.detect_outliers(column, method, threshold)
        if count == 0 or lower is None or upper is None:
            return 0

        desc = f"Handled {count} outliers in '{column}' using method='{method}', action='{action}' (bounds: [{lower:.2f}, {upper:.2f}])."
        self._record_state(
            action_type="handle_outliers",
            columns=[column],
            params={"method": method, "threshold": threshold, "action": action, "lower": lower, "upper": upper},
            description=desc,
        )

        if action == "cap":
            self.current_df[column] = self.current_df[column].clip(lower=lower, upper=upper)
        elif action == "remove":
            self.current_df = self.current_df[~mask].reset_index(drop=True)

        self._finalize_entry()
        return count

    def drop_columns(self, columns: List[str], reason: str = "user_choice") -> List[str]:
        """Drops specified columns from the dataset."""
        valid_cols = [c for c in columns if c in self.current_df.columns]
        if not valid_cols:
            return []

        desc = f"Dropped {len(valid_cols)} columns ({', '.join(valid_cols)}) [reason: {reason}]."
        self._record_state(
            action_type="drop_columns",
            columns=valid_cols,
            params={"columns": valid_cols, "reason": reason},
            description=desc,
        )
        self.current_df = self.current_df.drop(columns=valid_cols)
        self._finalize_entry()
        return valid_cols

    def convert_dtype(self, column: str, target_type: str) -> bool:
        """
        Converts column data type ('numeric', 'categorical', 'datetime', 'boolean').
        """
        if column not in self.current_df.columns:
            return False

        desc = f"Converted column '{column}' to '{target_type}'."
        self._record_state(
            action_type="convert_dtype",
            columns=[column],
            params={"target_type": target_type},
            description=desc,
        )

        try:
            if target_type == "numeric":
                self.current_df[column] = pd.to_numeric(self.current_df[column], errors="coerce")
            elif target_type == "categorical":
                self.current_df[column] = self.current_df[column].astype(str)
            elif target_type == "datetime":
                self.current_df[column] = pd.to_datetime(self.current_df[column], errors="coerce")
            elif target_type == "boolean":
                self.current_df[column] = self.current_df[column].astype(bool)
            self._finalize_entry()
            return True
        except Exception as e:
            # Revert if failed
            self.undo()
            raise ValueError(f"Failed to convert column '{column}' to {target_type}: {e}")

    @staticmethod
    def replay(original_df: pd.DataFrame, cleaning_log: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Replays an entire sequence of logged cleaning actions deterministically onto any DataFrame.
        """
        cleaner = DataCleaner(original_df)
        for step in cleaning_log:
            action = step.get("action")
            params = step.get("params", {})
            columns = step.get("columns", [])

            if action == "remove_duplicates":
                cleaner.handle_duplicates(subset=params.get("subset"), keep=params.get("keep", "first"))
            elif action == "handle_missing":
                cleaner.handle_missing(
                    columns=columns,
                    strategy=params.get("strategy", "mean"),
                    fill_value=params.get("fill_value"),
                )
            elif action == "handle_outliers":
                for col in columns:
                    cleaner.handle_outliers(
                        column=col,
                        method=params.get("method", "iqr"),
                        threshold=params.get("threshold", 1.5),
                        action=params.get("action", "cap"),
                    )
            elif action == "drop_columns":
                cleaner.drop_columns(columns=params.get("columns", columns), reason=params.get("reason", "replay"))
            elif action == "convert_dtype":
                for col in columns:
                    cleaner.convert_dtype(col, params.get("target_type", "categorical"))

        return cleaner.current_df
