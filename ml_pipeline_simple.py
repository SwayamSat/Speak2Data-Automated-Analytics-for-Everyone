"""
Simplified ML Pipeline module for Speak2Data platform.
Handles basic data analysis without complex ML dependencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimpleMLPipeline:
    """Simplified machine learning pipeline for business data analysis."""
    
    def __init__(self):
        """Initialize ML pipeline with default configurations."""
        self.model = None
        self.feature_columns = []
        self.target_column = None
        self.problem_type = None
        self.metrics = {}
        
    def analyze_data(self, data: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Analyze data and determine appropriate ML approach.
        
        Args:
            data: DataFrame containing the data
            target_column: Name of target column (if supervised learning)
            
        Returns:
            Dictionary containing analysis results and recommendations
        """
        analysis = {
            "data_shape": data.shape,
            "missing_values": data.isnull().sum().to_dict(),
            "data_types": data.dtypes.to_dict(),
            "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object']).columns.tolist(),
            "recommendations": []
        }
        
        # Check for missing values
        missing_pct = (data.isnull().sum() / len(data)) * 100
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            analysis["recommendations"].append(f"Consider dropping columns with >50% missing values: {high_missing}")
        
        # Check for constant columns
        constant_cols = data.columns[data.nunique() <= 1].tolist()
        if constant_cols:
            analysis["recommendations"].append(f"Remove constant columns: {constant_cols}")
        
        # Determine problem type if target column is specified
        if target_column and target_column in data.columns:
            target_data = data[target_column].dropna()
            unique_values = target_data.nunique()
            
            if unique_values <= 10:
                analysis["problem_type"] = "classification"
                analysis["target_classes"] = target_data.unique().tolist()
            else:
                analysis["problem_type"] = "regression"
                analysis["target_stats"] = {
                    "mean": target_data.mean(),
                    "std": target_data.std(),
                    "min": target_data.min(),
                    "max": target_data.max()
                }
        
        return analysis
    
    def prepare_data(self, data: pd.DataFrame, target_column: str, 
                    feature_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for machine learning.
        
        Args:
            data: DataFrame containing the data
            target_column: Name of target column
            feature_columns: List of feature column names (if None, auto-select)
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Auto-select features if not provided
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        # Remove rows with missing target values
        clean_data = data.dropna(subset=[target_column])
        
        # Prepare features
        features_df = clean_data[feature_columns].copy()
        target_series = clean_data[target_column]
        
        # Handle missing values in features
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        categorical_features = features_df.select_dtypes(include=['object']).columns
        
        # Fill missing numeric values with median
        for col in numeric_features:
            if features_df[col].isnull().any():
                features_df[col].fillna(features_df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        for col in categorical_features:
            if features_df[col].isnull().any():
                features_df[col].fillna(features_df[col].mode()[0], inplace=True)
        
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        return features_df, target_series
    
    def simple_linear_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Simple linear regression using numpy.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing model results
        """
        try:
            # Add intercept term
            X_with_intercept = np.column_stack([np.ones(len(X)), X.values])
            
            # Calculate coefficients using normal equation
            coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y.values
            
            # Make predictions
            predictions = X_with_intercept @ coefficients
            
            # Calculate R-squared
            ss_res = np.sum((y.values - predictions) ** 2)
            ss_tot = np.sum((y.values - np.mean(y.values)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y.values - predictions) ** 2))
            
            return {
                "coefficients": coefficients,
                "predictions": predictions,
                "r2_score": r2,
                "rmse": rmse,
                "model_type": "linear_regression"
            }
            
        except Exception as e:
            return {"error": str(e), "model_type": "linear_regression"}
    
    def simple_classification(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Simple classification using basic rules.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing model results
        """
        try:
            # Get unique classes
            classes = y.unique()
            
            # Calculate class means for each feature
            class_means = {}
            for cls in classes:
                class_data = X[y == cls]
                class_means[cls] = class_data.mean().to_dict()
            
            # Simple prediction: find closest class mean
            predictions = []
            for idx, row in X.iterrows():
                distances = {}
                for cls, means in class_means.items():
                    distance = np.sqrt(np.sum([(row[col] - means[col])**2 for col in X.columns]))
                    distances[cls] = distance
                
                predicted_class = min(distances, key=distances.get)
                predictions.append(predicted_class)
            
            # Calculate accuracy
            accuracy = np.mean([pred == actual for pred, actual in zip(predictions, y)])
            
            return {
                "class_means": class_means,
                "predictions": predictions,
                "accuracy": accuracy,
                "model_type": "simple_classification"
            }
            
        except Exception as e:
            return {"error": str(e), "model_type": "simple_classification"}
    
    def train_model(self, features_df: pd.DataFrame, target_series: pd.Series, 
                   problem_type: str) -> Dict[str, Any]:
        """Train machine learning model.
        
        Args:
            features_df: DataFrame containing features
            target_series: Series containing target values
            problem_type: Type of ML problem ('classification', 'regression', 'clustering')
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            if problem_type == 'regression':
                results = self.simple_linear_regression(features_df, target_series)
            elif problem_type == 'classification':
                results = self.simple_classification(features_df, target_series)
            else:
                return {
                    "error": f"Unsupported problem type: {problem_type}",
                    "training_successful": False
                }
            
            if "error" in results:
                return {
                    "error": results["error"],
                    "training_successful": False
                }
            
            # Store model and results
            self.model = results
            self.problem_type = problem_type
            
            # Extract metrics
            metrics = {}
            if "r2_score" in results:
                metrics["r2_score"] = results["r2_score"]
            if "rmse" in results:
                metrics["rmse"] = results["rmse"]
            if "accuracy" in results:
                metrics["accuracy"] = results["accuracy"]
            
            self.metrics = metrics
            
            return {
                "model": results,
                "metrics": metrics,
                "predictions": results.get("predictions", []),
                "training_successful": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "training_successful": False
            }
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model.
        
        Args:
            features_df: DataFrame containing features for prediction
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("No model trained. Call train_model first.")
        
        if self.problem_type == 'regression' and "coefficients" in self.model:
            # Linear regression prediction
            X_with_intercept = np.column_stack([np.ones(len(features_df)), features_df.values])
            return X_with_intercept @ self.model["coefficients"]
        elif self.problem_type == 'classification' and "class_means" in self.model:
            # Simple classification prediction
            predictions = []
            for idx, row in features_df.iterrows():
                distances = {}
                for cls, means in self.model["class_means"].items():
                    distance = np.sqrt(np.sum([(row[col] - means[col])**2 for col in features_df.columns]))
                    distances[cls] = distance
                predicted_class = min(distances, key=distances.get)
                predictions.append(predicted_class)
            return np.array(predictions)
        else:
            raise ValueError("Model not properly trained or unsupported type")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained model.
        
        Returns:
            Dictionary containing model summary
        """
        if self.model is None:
            return {"error": "No model trained"}
        
        summary = {
            "problem_type": self.problem_type,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "metrics": self.metrics,
            "model_type": self.model.get("model_type", "unknown")
        }
        
        return summary
    
    def explain_prediction(self, features_df: pd.DataFrame, prediction: Any) -> str:
        """Generate explanation for a specific prediction.
        
        Args:
            features_df: DataFrame containing features
            prediction: Prediction value
            
        Returns:
            Human-readable explanation
        """
        if self.model is None:
            return "No model available for explanation"
        
        try:
            if self.problem_type == 'classification':
                return f"The model predicts this record belongs to class: {prediction}"
            elif self.problem_type == 'regression':
                return f"The model predicts a value of: {prediction:.2f}"
            else:
                return f"Prediction: {prediction}"
        except Exception as e:
            return f"Unable to explain prediction: {str(e)}"
