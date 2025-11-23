"""
Data preprocessing pipeline for the Speak2Data system.
Handles cleaning, encoding, scaling, and train/test splitting.
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from utils import infer_column_type, get_column_statistics
from config import config


@dataclass
class PreprocessingMetadata:
    """Metadata about the preprocessing operations."""
    original_columns: List[str]
    target_column: Optional[str]
    feature_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    dropped_columns: List[str]
    column_types: Dict[str, str]
    encoders: Dict[str, Any]
    scaler: Optional[Any]
    imputers: Dict[str, Any]
    train_size: int
    test_size: int


@dataclass
class DataBundle:
    """Container for preprocessed data."""
    X_train: Optional[pd.DataFrame]
    X_test: Optional[pd.DataFrame]
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]
    X: Optional[pd.DataFrame]  # For unsupervised tasks
    original_df: pd.DataFrame
    metadata: PreprocessingMetadata


def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify column types in a DataFrame.
    
    Returns:
        Dict with keys: numeric, categorical, datetime, text, boolean
    """
    types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "boolean": []
    }
    
    for col in df.columns:
        col_type = infer_column_type(df[col])
        types[col_type].append(col)
    
    return types


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    numeric_strategy: str = "mean",
    categorical_strategy: str = "most_frequent"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing values in a DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Overall strategy - "drop", "impute", or "keep"
        numeric_strategy: Strategy for numeric columns - "mean", "median", "most_frequent"
        categorical_strategy: Strategy for categorical columns - "most_frequent", "constant"
        
    Returns:
        Tuple of (cleaned DataFrame, dict of imputers)
    """
    imputers = {}
    original_rows = len(df)
    
    if strategy == "drop":
        # Drop rows with any missing values
        df = df.dropna()
        
        # Check if we removed too many rows
        if len(df) == 0:
            raise ValueError(f"All {original_rows} rows contain missing values. Consider using strategy='impute' instead of 'drop'.")
        
        if len(df) < original_rows * 0.1:  # Lost more than 90% of data
            print(f"Warning: Dropped {original_rows - len(df)} rows ({(1 - len(df)/original_rows)*100:.1f}% of data) due to missing values. Consider using imputation.")
    
    elif strategy == "impute":
        column_types = identify_column_types(df)
        
        # Impute numeric columns
        if column_types["numeric"]:
            numeric_imputer = SimpleImputer(strategy=numeric_strategy)
            df[column_types["numeric"]] = numeric_imputer.fit_transform(df[column_types["numeric"]])
            imputers["numeric"] = numeric_imputer
        
        # Impute categorical columns
        if column_types["categorical"]:
            categorical_imputer = SimpleImputer(strategy=categorical_strategy, fill_value="Unknown")
            df[column_types["categorical"]] = categorical_imputer.fit_transform(df[column_types["categorical"]])
            imputers["categorical"] = categorical_imputer
        
        # For boolean, fill with mode
        if column_types["boolean"]:
            for col in column_types["boolean"]:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else False)
    
    return df, imputers


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    method: str = "onehot",
    max_categories: int = 10
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical column names
        method: Encoding method - "onehot" or "label"
        max_categories: Maximum categories for one-hot encoding
        
    Returns:
        Tuple of (encoded DataFrame, dict of encoders)
    """
    encoders = {}
    
    for col in categorical_columns:
        if col not in df.columns:
            continue
        
        n_categories = df[col].nunique()
        
        if method == "onehot" and n_categories <= max_categories:
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoders[col] = {"type": "onehot", "columns": list(dummies.columns)}
        
        else:
            # Label encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = {"type": "label", "encoder": le}
    
    return df, encoders


def scale_features(
    df: pd.DataFrame,
    numeric_columns: List[str],
    method: str = "standard"
) -> Tuple[pd.DataFrame, Optional[Any]]:
    """
    Scale numeric features.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names
        method: Scaling method - "standard", "minmax", "robust", or "none"
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    if method == "none" or not numeric_columns:
        return df, None
    
    # Filter to existing columns
    numeric_columns = [col for col in numeric_columns if col in df.columns]
    
    if not numeric_columns:
        return df, None
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        return df, None
    
    # Validate before scaling
    if len(df) == 0:
        raise ValueError("Cannot scale features: dataframe has 0 rows")
    
    if len(numeric_columns) == 0 or df[numeric_columns].shape[1] == 0:
        # No numeric columns to scale
        return df, None
    
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df, scaler


def preprocess_for_descriptive(df: pd.DataFrame) -> DataBundle:
    """
    Minimal preprocessing for descriptive analytics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataBundle with preprocessed data
    """
    # Just handle missing values minimally
    df_clean, imputers = handle_missing_values(df, strategy="keep")
    
    column_types = identify_column_types(df_clean)
    
    metadata = PreprocessingMetadata(
        original_columns=list(df.columns),
        target_column=None,
        feature_columns=list(df.columns),
        numeric_columns=column_types["numeric"],
        categorical_columns=column_types["categorical"],
        datetime_columns=column_types["datetime"],
        dropped_columns=[],
        column_types={col: infer_column_type(df[col]) for col in df.columns},
        encoders={},
        scaler=None,
        imputers=imputers,
        train_size=len(df),
        test_size=0
    )
    
    return DataBundle(
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        X=df_clean,
        original_df=df,
        metadata=metadata
    )


def preprocess_for_supervised(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    task_type: str = "classification"
) -> DataBundle:
    """
    Preprocess data for supervised learning (classification or regression).
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion of data for testing
        random_state: Random seed
        task_type: "classification" or "regression"
        
    Returns:
        DataBundle with train/test splits
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Check minimum data size
    min_samples = max(10, int(1 / test_size) + 2)  # At least enough for train/test split
    if len(df) < min_samples:
        raise ValueError(f"Not enough data for supervised learning. Need at least {min_samples} rows, got {len(df)}")
    
    # Separate target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    print(f"DEBUG: After separating target - X shape: {X.shape}, X columns: {list(X.columns)}")
    print(f"DEBUG: y shape: {y.shape}, y name: {y.name}")
    
    # Check if we have any features
    if X.shape[1] == 0:
        raise ValueError(f"No feature columns found after removing target '{target_column}'. The dataset only contains the target column.")
    
    # Check if target has enough variation
    if y.nunique() < 2:
        raise ValueError(f"Target column '{target_column}' has insufficient variation (only {y.nunique()} unique value(s))")
    
    original_columns = list(df.columns)
    
    # Handle missing values with fallback
    try:
        X, imputers = handle_missing_values(X, strategy=config.missing_strategy)
    except ValueError as e:
        if "All" in str(e) and "rows contain missing values" in str(e):
            # Fallback to imputation
            print(f"Warning: {e}. Falling back to imputation strategy.")
            X, imputers = handle_missing_values(X, strategy="impute")
        else:
            raise
    
    # Check if we still have data after handling missing values
    if len(X) == 0:
        raise ValueError(f"All rows were removed during missing value handling. Original data had {len(df)} rows. Try using strategy='impute'.")
    
    y = y[X.index]  # Align with X after dropping
    
    # Identify column types
    column_types = identify_column_types(X)
    
    # Handle text columns intelligently
    dropped_columns = []
    print(f"DEBUG: Before text handling - X shape: {X.shape}, columns: {list(X.columns)}")
    print(f"DEBUG: Column types - text: {column_types['text']}, categorical: {column_types['categorical']}, numeric: {column_types['numeric']}")
    
    if column_types["text"]:
        # Check if we have other features
        non_text_cols = [col for col in X.columns if col not in column_types["text"]]
        
        if len(non_text_cols) > 0:
            # We have other features, safe to drop text columns
            X = X.drop(columns=column_types["text"])
            dropped_columns = column_types["text"]
            print(f"Dropped {len(dropped_columns)} text columns with high cardinality")
        else:
            # Text columns are the only features - try to use them
            print(f"Warning: Only text columns available. Attempting to encode them as categorical.")
            print(f"DEBUG: Moving text columns to categorical: {column_types['text']}")
            # Treat high-cardinality text as categorical (will be label-encoded)
            column_types["categorical"].extend(column_types["text"])
            column_types["text"] = []
            print(f"DEBUG: After move - text: {column_types['text']}, categorical: {column_types['categorical']}")
    
    # Check if we still have columns
    print(f"DEBUG: After text handling - X shape: {X.shape}, columns: {list(X.columns)}")
    if X.shape[1] == 0:
        error_msg = (
            f"No features remaining after preprocessing.\n"
            f"Original dataframe had {len(df.columns)} columns including target.\n"
            f"After separating target '{target_column}', had {len(df.columns)-1} feature columns.\n"
            f"Columns dropped as text: {dropped_columns}\n"
            f"This suggests all non-target columns were classified as high-cardinality text and dropped.\n"
            f"Try using a dataset with numeric or categorical features."
        )
        raise ValueError(error_msg)
    
    # Update column types after dropping
    column_types = identify_column_types(X)
    
    # Encode categorical features
    X, encoders = encode_categorical_features(
        X,
        column_types["categorical"],
        method=config.categorical_encoding
    )
    
    # Convert boolean to int
    for col in column_types["boolean"]:
        if col in X.columns:
            X[col] = X[col].astype(int)
    
    # Handle datetime columns (extract features)
    for col in column_types["datetime"]:
        if col in X.columns:
            X[col] = pd.to_datetime(X[col])
            X[f"{col}_year"] = X[col].dt.year
            X[f"{col}_month"] = X[col].dt.month
            X[f"{col}_day"] = X[col].dt.day
            X[f"{col}_dayofweek"] = X[col].dt.dayofweek
            X = X.drop(columns=[col])
    
    # Scale numeric features
    numeric_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    
    # Final validation before scaling
    if len(X) == 0:
        raise ValueError(f"No samples remaining after preprocessing. Check your data quality and missing value strategy.")
    
    if X.shape[1] == 0:
        raise ValueError("No features remaining after preprocessing. Ensure your data has numeric or categorical columns.")
    
    X, scaler = scale_features(X, numeric_cols, method=config.scaling_method)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task_type == "classification" and y.nunique() > 1 and y.nunique() < len(y) * 0.5 else None
    )
    
    metadata = PreprocessingMetadata(
        original_columns=original_columns,
        target_column=target_column,
        feature_columns=list(X.columns),
        numeric_columns=numeric_cols,
        categorical_columns=column_types["categorical"],
        datetime_columns=column_types["datetime"],
        dropped_columns=dropped_columns,
        column_types={col: infer_column_type(df[col]) for col in df.columns if col in df.columns},
        encoders=encoders,
        scaler=scaler,
        imputers=imputers,
        train_size=len(X_train),
        test_size=len(X_test)
    )
    
    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X=X,
        original_df=df,
        metadata=metadata
    )


def preprocess_for_unsupervised(
    df: pd.DataFrame,
    task_type: str = "clustering"
) -> DataBundle:
    """
    Preprocess data for unsupervised learning (clustering, PCA, etc.).
    
    Args:
        df: Input DataFrame
        task_type: Type of unsupervised task
        
    Returns:
        DataBundle with preprocessed features
    """
    original_columns = list(df.columns)
    
    # Handle missing values with fallback
    try:
        df, imputers = handle_missing_values(df, strategy=config.missing_strategy)
    except ValueError as e:
        if "All" in str(e) and "rows contain missing values" in str(e):
            # Fallback to imputation
            print(f"Warning: {e}. Falling back to imputation strategy.")
            df, imputers = handle_missing_values(df, strategy="impute")
        else:
            raise
    
    # Validate we still have data
    if len(df) == 0:
        raise ValueError("All rows were removed during preprocessing. Try using strategy='impute'.")
    
    # Identify column types
    column_types = identify_column_types(df)
    
    # Drop text columns intelligently
    if column_types["text"]:
        # Check if there are non-text columns
        non_text_cols = [col for col in df.columns if col not in column_types["text"]]
        if len(non_text_cols) > 0:
            # Safe to drop text columns
            df = df.drop(columns=column_types["text"])
            dropped_columns = column_types["text"]
            print(f"Dropped text columns for unsupervised learning: {column_types['text']}")
        else:
            # Only text columns exist - treat them as categorical
            print(f"Text columns are the only features. Encoding them as categorical: {column_types['text']}")
            column_types["categorical"].extend(column_types["text"])
            column_types["text"] = []
            dropped_columns = []
    else:
        dropped_columns = []
    
    # Update column types
    column_types = identify_column_types(df)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(
        df,
        column_types["categorical"],
        method="label"  # Use label encoding for clustering
    )
    
    # Convert boolean to int
    for col in column_types["boolean"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Handle datetime columns
    for col in column_types["datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df[f"{col}_timestamp"] = df[col].astype(np.int64) // 10**9
            df = df.drop(columns=[col])
    
    # Scale all features (important for clustering)
    numeric_cols = list(df.columns)
    df, scaler = scale_features(df, numeric_cols, method="standard")
    
    metadata = PreprocessingMetadata(
        original_columns=original_columns,
        target_column=None,
        feature_columns=list(df.columns),
        numeric_columns=numeric_cols,
        categorical_columns=column_types["categorical"],
        datetime_columns=column_types["datetime"],
        dropped_columns=dropped_columns,
        column_types={col: infer_column_type(df[col]) for col in df.columns if col in df.columns},
        encoders=encoders,
        scaler=scaler,
        imputers=imputers,
        train_size=len(df),
        test_size=0
    )
    
    return DataBundle(
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        X=df,
        original_df=df,
        metadata=metadata
    )


def preprocess_data(
    df: pd.DataFrame,
    task_type: str,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> DataBundle:
    """
    Main preprocessing function that routes to appropriate preprocessing method.
    
    Args:
        df: Input DataFrame
        task_type: Type of task
        target_column: Target column for supervised tasks
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        DataBundle with preprocessed data
    """
    if task_type in ["descriptive_analytics", "aggregation", "comparison"]:
        return preprocess_for_descriptive(df)
    
    elif task_type in ["classification", "regression", "time_series_forecast"]:
        if not target_column:
            raise ValueError(f"Target column required for {task_type}")
        return preprocess_for_supervised(df, target_column, test_size, random_state, task_type)
    
    elif task_type in ["clustering", "correlation_analysis"]:
        return preprocess_for_unsupervised(df, task_type)
    
    else:
        # Default to descriptive
        return preprocess_for_descriptive(df)
