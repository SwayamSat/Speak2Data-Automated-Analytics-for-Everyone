"""
Machine Learning pipeline for the Speak2Data system.
Supports multiple task types with automated model comparison.
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import warnings

from data_preprocessing import DataBundle
from config import config

warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """Results from a single model."""
    model_name: str
    model: Any
    metrics: Dict[str, float]
    predictions: Any
    feature_importances: Optional[Dict[str, float]] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class PipelineResult:
    """Complete pipeline results."""
    task_type: str
    best_model_name: str
    best_model: Any
    all_results: List[ModelResult]
    metrics_comparison: pd.DataFrame
    predictions: Any
    feature_importances: Optional[Dict[str, float]] = None
    additional_data: Optional[Dict[str, Any]] = None


def get_feature_importances(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
    """
    Extract feature importances from a model if available.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if len(importances.shape) > 1:
                importances = importances[0]
        else:
            return None
        
        # Create dictionary
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    except Exception as e:
        print(f"Warning: Could not extract feature importances: {e}")
        return None


def train_classification_models(data_bundle: DataBundle) -> List[ModelResult]:
    """
    Train multiple classification models and evaluate.
    
    Args:
        data_bundle: Preprocessed data
        
    Returns:
        List of ModelResult objects
    """
    X_train = data_bundle.X_train
    X_test = data_bundle.X_test
    y_train = data_bundle.y_train
    y_test = data_bundle.y_test
    
    results = []
    
    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=config.random_state),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=config.random_state),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=config.random_state)
    }
    
    for model_name, model in models.items():
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
            
            # ROC AUC for binary classification
            if len(np.unique(y_train)) == 2 and y_prob is not None:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                except:
                    pass
            
            # Feature importances
            feature_importances = get_feature_importances(model, list(X_train.columns))
            
            results.append(ModelResult(
                model_name=model_name,
                model=model,
                metrics=metrics,
                predictions=y_pred,
                feature_importances=feature_importances,
                additional_data={
                    "y_true": y_test,
                    "y_prob": y_prob,
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
                }
            ))
        
        except Exception as e:
            print(f"Warning: {model_name} failed: {e}")
    
    return results


def train_regression_models(data_bundle: DataBundle) -> List[ModelResult]:
    """
    Train multiple regression models and evaluate.
    
    Args:
        data_bundle: Preprocessed data
        
    Returns:
        List of ModelResult objects
    """
    X_train = data_bundle.X_train
    X_test = data_bundle.X_test
    y_train = data_bundle.y_train
    y_test = data_bundle.y_test
    
    results = []
    
    # Define models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=config.random_state),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=config.random_state)
    }
    
    for model_name, model in models.items():
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "mse": float(mean_squared_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": float(r2_score(y_test, y_pred))
            }
            
            # Feature importances
            feature_importances = get_feature_importances(model, list(X_train.columns))
            
            results.append(ModelResult(
                model_name=model_name,
                model=model,
                metrics=metrics,
                predictions=y_pred,
                feature_importances=feature_importances,
                additional_data={
                    "y_true": y_test,
                    "residuals": (y_test - y_pred).tolist()
                }
            ))
        
        except Exception as e:
            print(f"Warning: {model_name} failed: {e}")
    
    return results


def train_clustering_models(data_bundle: DataBundle) -> List[ModelResult]:
    """
    Train multiple clustering models and evaluate.
    
    Args:
        data_bundle: Preprocessed data
        
    Returns:
        List of ModelResult objects
    """
    X = data_bundle.X
    
    results = []
    
    # Try different K values for KMeans
    k_min, k_max = config.kmeans_k_range
    best_kmeans = None
    best_silhouette = -1
    
    for k in range(k_min, k_max + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=config.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            silhouette = silhouette_score(X, labels)
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_kmeans = (k, kmeans, labels, silhouette)
        
        except Exception as e:
            print(f"Warning: KMeans with k={k} failed: {e}")
    
    # Add best KMeans result
    if best_kmeans:
        k, model, labels, silhouette = best_kmeans
        
        results.append(ModelResult(
            model_name=f"KMeans_k{k}",
            model=model,
            metrics={
                "silhouette_score": float(silhouette),
                "n_clusters": k,
                "inertia": float(model.inertia_)
            },
            predictions=labels,
            additional_data={
                "cluster_centers": model.cluster_centers_.tolist(),
                "cluster_sizes": {int(i): int(np.sum(labels == i)) for i in range(k)}
            }
        ))
    
    # Try DBSCAN
    try:
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
            
            results.append(ModelResult(
                model_name="DBSCAN",
                model=dbscan,
                metrics={
                    "silhouette_score": float(silhouette),
                    "n_clusters": n_clusters,
                    "n_noise": int(np.sum(labels == -1))
                },
                predictions=labels,
                additional_data={
                    "cluster_sizes": {int(i): int(np.sum(labels == i)) for i in set(labels)}
                }
            ))
    
    except Exception as e:
        print(f"Warning: DBSCAN failed: {e}")
    
    return results


def train_time_series_models(data_bundle: DataBundle) -> List[ModelResult]:
    """
    Train simple time series models.
    
    For simplicity, we use RandomForest with lagged features.
    
    Args:
        data_bundle: Preprocessed data
        
    Returns:
        List of ModelResult objects
    """
    # For now, treat it like regression
    return train_regression_models(data_bundle)


def select_best_model(results: List[ModelResult], task_type: str) -> ModelResult:
    """
    Select the best model based on task-specific metrics.
    
    Args:
        results: List of ModelResult objects
        task_type: Type of task
        
    Returns:
        Best ModelResult
    """
    if not results:
        raise ValueError("No models to select from")
    
    if task_type == "classification":
        # Select by F1 score
        best = max(results, key=lambda r: r.metrics.get("f1_score", 0))
    
    elif task_type == "regression":
        # Select by R²
        best = max(results, key=lambda r: r.metrics.get("r2", -float('inf')))
    
    elif task_type == "clustering":
        # Select by silhouette score
        best = max(results, key=lambda r: r.metrics.get("silhouette_score", -1))
    
    else:
        # Default to first model
        best = results[0]
    
    return best


def run_pipeline(
    task_type: str,
    data_bundle: DataBundle,
    pipeline_config: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """
    Main ML pipeline execution.
    
    Args:
        task_type: Type of task
        data_bundle: Preprocessed data
        pipeline_config: Optional configuration overrides
        
    Returns:
        PipelineResult with all model results
    """
    if task_type in ["descriptive_analytics", "aggregation", "comparison", "correlation_analysis"]:
        # No ML training needed
        return PipelineResult(
            task_type=task_type,
            best_model_name="None",
            best_model=None,
            all_results=[],
            metrics_comparison=pd.DataFrame(),
            predictions=None,
            additional_data={"data": data_bundle.X}
        )
    
    # Train models based on task type
    if task_type == "classification":
        results = train_classification_models(data_bundle)
    
    elif task_type == "regression":
        results = train_regression_models(data_bundle)
    
    elif task_type == "clustering":
        results = train_clustering_models(data_bundle)
    
    elif task_type == "time_series_forecast":
        results = train_time_series_models(data_bundle)
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    if not results:
        raise RuntimeError("No models trained successfully")
    
    # Select best model
    best_result = select_best_model(results, task_type)
    
    # Create metrics comparison DataFrame
    metrics_data = []
    for result in results:
        row = {"model": result.model_name}
        row.update(result.metrics)
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    
    return PipelineResult(
        task_type=task_type,
        best_model_name=best_result.model_name,
        best_model=best_result.model,
        all_results=results,
        metrics_comparison=metrics_df,
        predictions=best_result.predictions,
        feature_importances=best_result.feature_importances,
        additional_data=best_result.additional_data
    )
