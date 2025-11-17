"""
Visualization module for the Speak2Data system.
Creates interactive Plotly visualizations for various task types.
"""
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_preprocessing import DataBundle
from ml_pipeline import PipelineResult


def plot_dataframe_overview(df: pd.DataFrame, title: str = "Data Overview") -> go.Figure:
    """
    Create an overview visualization of a DataFrame.
    
    Args:
        df: DataFrame to visualize
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Get basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        # No numeric columns, show a message
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric columns to visualize",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title)
        return fig
    
    # Create subplots for histograms
    n_cols = min(len(numeric_cols), 4)
    fig = make_subplots(
        rows=(len(numeric_cols) + n_cols - 1) // n_cols,
        cols=n_cols,
        subplot_titles=[f"{col}" for col in numeric_cols[:12]]  # Limit to 12
    )
    
    for idx, col in enumerate(list(numeric_cols)[:12]):
        row = idx // n_cols + 1
        col_idx = idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_idx
        )
    
    fig.update_layout(
        title=title,
        height=300 * ((len(numeric_cols) + n_cols - 1) // n_cols),
        showlegend=False
    )
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Create a correlation matrix heatmap.
    
    Args:
        df: DataFrame with numeric columns
        
    Returns:
        Plotly figure
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 numeric columns for correlation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        height=600
    )
    
    return fig


def plot_classification_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None) -> go.Figure:
    """
    Create a confusion matrix heatmap for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional label names
        
    Returns:
        Plotly figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [str(i) for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500,
        width=500
    )
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
    """
    Create ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        
    Returns:
        Plotly figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        width=500,
        showlegend=True
    )
    
    return fig


def plot_regression_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Create predicted vs actual scatter plot for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(size=8, opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Predicted vs Actual Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=500,
        width=500,
        showlegend=True
    )
    
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Create residual plot for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Plotly figure
    """
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(size=8, opacity=0.6)
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        height=500,
        width=600,
        showlegend=True
    )
    
    return fig


def plot_feature_importances(importances: Dict[str, float], top_n: int = 10) -> go.Figure:
    """
    Create feature importance bar chart.
    
    Args:
        importances: Dictionary mapping feature names to importance scores
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    # Get top N features
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, scores = zip(*sorted_features) if sorted_features else ([], [])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(scores),
        y=list(features),
        orientation='h',
        marker=dict(color='skyblue')
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def plot_clustering_results(X: pd.DataFrame, labels: np.ndarray, method: str = "PCA") -> go.Figure:
    """
    Create 2D visualization of clustering results.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        method: Dimensionality reduction method ("PCA" or "TSNE")
        
    Returns:
        Plotly figure
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Reduce to 2D
    if method == "PCA":
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        x_label = f"PC1 ({reducer.explained_variance_ratio_[0]:.1%})"
        y_label = f"PC2 ({reducer.explained_variance_ratio_[1]:.1%})"
    else:
        reducer = TSNE(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        x_label = "Dimension 1"
        y_label = "Dimension 2"
    
    # Create DataFrame
    plot_df = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'cluster': labels.astype(str)
    })
    
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='cluster',
        title=f"Clustering Results ({method})",
        labels={'x': x_label, 'y': y_label}
    )
    
    fig.update_layout(height=600, width=800)
    
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart comparing model metrics.
    
    Args:
        metrics_df: DataFrame with model names and metrics
        
    Returns:
        Plotly figure
    """
    if metrics_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Get metric columns (exclude 'model' column)
    metric_cols = [col for col in metrics_df.columns if col != 'model']
    
    if not metric_cols:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create grouped bar chart
    fig = go.Figure()
    
    for metric in metric_cols:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['model'],
            y=metrics_df[metric],
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=500,
        legend_title="Metrics"
    )
    
    return fig


def create_visualizations(
    task_type: str,
    data_bundle: DataBundle,
    pipeline_result: Optional[PipelineResult] = None
) -> Dict[str, go.Figure]:
    """
    Create all relevant visualizations based on task type.
    
    Args:
        task_type: Type of task
        data_bundle: Preprocessed data
        pipeline_result: Optional pipeline results
        
    Returns:
        Dictionary mapping visualization names to Plotly figures
    """
    figures = {}
    
    # Data overview (always show)
    if data_bundle.X is not None:
        figures["data_overview"] = plot_dataframe_overview(data_bundle.X, "Data Distribution")
    
    # Correlation matrix for numeric data
    if data_bundle.X is not None and len(data_bundle.X.select_dtypes(include=[np.number]).columns) > 1:
        figures["correlation"] = plot_correlation_matrix(data_bundle.X)
    
    # Task-specific visualizations
    if pipeline_result and pipeline_result.all_results:
        
        if task_type == "classification":
            # Confusion matrix
            best_result = next((r for r in pipeline_result.all_results if r.model_name == pipeline_result.best_model_name), None)
            if best_result and best_result.additional_data:
                y_true = best_result.additional_data.get("y_true")
                y_pred = best_result.predictions
                
                if y_true is not None and y_pred is not None:
                    figures["confusion_matrix"] = plot_classification_confusion_matrix(y_true, y_pred)
                
                # ROC curve for binary classification
                y_prob = best_result.additional_data.get("y_prob")
                if y_prob is not None and len(np.unique(y_true)) == 2:
                    figures["roc_curve"] = plot_roc_curve(y_true, y_prob[:, 1])
        
        elif task_type == "regression":
            # Predicted vs actual
            best_result = next((r for r in pipeline_result.all_results if r.model_name == pipeline_result.best_model_name), None)
            if best_result and best_result.additional_data:
                y_true = best_result.additional_data.get("y_true")
                y_pred = best_result.predictions
                
                if y_true is not None and y_pred is not None:
                    figures["predictions"] = plot_regression_predictions(y_true, y_pred)
                    figures["residuals"] = plot_residuals(y_true, y_pred)
        
        elif task_type == "clustering":
            # Clustering visualization
            if data_bundle.X is not None and pipeline_result.predictions is not None:
                figures["clusters"] = plot_clustering_results(data_bundle.X, pipeline_result.predictions)
        
        # Feature importances (if available)
        if pipeline_result.feature_importances:
            figures["feature_importances"] = plot_feature_importances(pipeline_result.feature_importances)
        
        # Metrics comparison
        if not pipeline_result.metrics_comparison.empty:
            figures["metrics_comparison"] = plot_metrics_comparison(pipeline_result.metrics_comparison)
    
    return figures
