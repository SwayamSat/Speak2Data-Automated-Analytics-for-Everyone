import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

class MLPipelineManager:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.models = {}
    
    def create_pipeline(self, data: pd.DataFrame, 
                       query_intent: Dict[str, Any]) -> Optional[Dict]:
        """Create and execute ML pipeline based on data and intent"""
        if data.empty:
            return None
        
        try:
            # Determine ML task type
            ml_task = self._determine_ml_task(data, query_intent)
            
            print(f"Determined ML task: {ml_task}")
            
            if ml_task == 'regression':
                return self._create_regression_pipeline(data, query_intent)
            elif ml_task == 'classification':
                return self._create_classification_pipeline(data, query_intent)
            elif ml_task == 'forecasting':
                return self._create_forecasting_pipeline(data, query_intent)
            elif ml_task == 'clustering':
                return self._create_clustering_pipeline(data, query_intent)
            else:
                return self._create_exploratory_analysis(data)
        
        except Exception as e:
            print(f"Error creating ML pipeline: {e}")
            return None
    
    def _determine_ml_task(self, data: pd.DataFrame, 
                          query_intent: Dict[str, Any]) -> str:
        """Determine what type of ML task is needed"""
        query_type = query_intent.get('query_type', '').lower()
        intent = query_intent.get('intent', '').lower()
        
        # Check for prediction keywords
        prediction_keywords = ['predict', 'forecast', 'estimate', 'future', 'trend']
        if any(keyword in intent for keyword in prediction_keywords):
            # Check if it's time series data
            date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
            if len(date_cols) > 0:
                return 'forecasting'
            else:
                return 'regression'
        
        # Check for classification keywords
        classification_keywords = ['classify', 'category', 'segment', 'group', 'cluster']
        if any(keyword in intent for keyword in classification_keywords):
            return 'classification'
        
        # Check for regression keywords
        regression_keywords = ['sales', 'revenue', 'amount', 'price', 'value', 'income']
        if any(keyword in intent for keyword in regression_keywords):
            return 'regression'
        
        # Default based on data characteristics
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            return 'regression'
        elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
            return 'classification'
        
        return 'exploratory'
    
    def _create_regression_pipeline(self, data: pd.DataFrame, 
                                   query_intent: Dict) -> Dict:
        """Create regression pipeline for numerical predictions"""
        results = {}
        
        # Identify target variable (typically the last numeric column or amount-related)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return self._create_exploratory_analysis(data)
        
        # Smart target selection
        target_col = self._select_target_column(numeric_cols, query_intent)
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        if len(feature_cols) == 0:
            return self._create_exploratory_analysis(data)
        
        # Prepare data
        X = data[feature_cols].dropna()
        y = data[target_col].dropna()
        
        # Ensure X and y have same indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 10:  # Not enough data for ML
            return self._create_exploratory_analysis(data)
        
        # Split data
        test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))  # Adaptive test size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        }
        
        best_model = None
        best_score = -np.inf
        model_results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                model_results[name] = {
                    'r2_score': r2,
                    'mse': mse,
                    'predictions': y_pred
                }
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if best_model is None:
            return self._create_exploratory_analysis(data)
        
        # Generate predictions
        future_predictions = best_model.predict(X_test_scaled)
        
        # Create visualizations
        visualizations = []
        
        # Actual vs Predicted plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=future_predictions,
            mode='markers',
            name='Predictions',
            text=[f'Actual: {a:.2f}<br>Predicted: {p:.2f}' 
                  for a, p in zip(y_test, future_predictions)],
            marker=dict(color='blue', size=8)
        ))
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        fig.update_layout(
            title=f'Actual vs Predicted {target_col}',
            xaxis_title=f'Actual {target_col}',
            yaxis_title=f'Predicted {target_col}',
            height=500
        )
        visualizations.append(fig)
        
        # Feature importance (for Random Forest)
        if 'Random Forest' in model_results and hasattr(models['Random Forest'], 'feature_importances_'):
            rf_model = models['Random Forest']
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(importance_df, x='importance', y='feature',
                        title='Feature Importance (Random Forest)',
                        orientation='h',
                        color='importance',
                        color_continuous_scale='viridis')
            fig.update_layout(height=400)
            visualizations.append(fig)
        
        # Prediction residuals
        residuals = y_test - future_predictions
        fig = px.histogram(residuals, title='Prediction Residuals Distribution',
                          nbins=20, color_discrete_sequence=['lightblue'])
        fig.update_layout(height=400)
        visualizations.append(fig)
        
        results = {
            'model_performance': {
                'best_model': type(best_model).__name__,
                'r2_score': best_score,
                'accuracy': max(0, best_score),  # Ensure non-negative
                'target_column': target_col,
                'feature_columns': feature_cols
            },
            'predictions': pd.DataFrame({
                'actual': y_test,
                'predicted': future_predictions,
                'residual': residuals
            }),
            'visualizations': visualizations,
            'model_results': model_results
        }
        
        return results
    
    def _select_target_column(self, numeric_cols: List[str], query_intent: Dict) -> str:
        """Intelligently select target column based on query intent"""
        intent = query_intent.get('intent', '').lower()
        
        # Priority keywords for target selection
        target_keywords = {
            'sales': ['sales', 'revenue', 'amount', 'total'],
            'price': ['price', 'cost'],
            'profit': ['profit', 'margin'],
            'quantity': ['quantity', 'count']
        }
        
        for keyword_group, keywords in target_keywords.items():
            for keyword in keywords:
                if keyword in intent:
                    for col in numeric_cols:
                        if keyword in col.lower():
                            return col
        
        # Default to last column or one with 'amount', 'total', 'sales' in name
        priority_cols = [col for col in numeric_cols 
                        if any(word in col.lower() for word in ['amount', 'total', 'sales', 'revenue'])]
        
        if priority_cols:
            return priority_cols[0]
        
        return numeric_cols[-1]  # Default to last numeric column
    
    def _create_classification_pipeline(self, data: pd.DataFrame, 
                                       query_intent: Dict) -> Dict:
        """Create classification pipeline"""
        results = {}
        
        # Find categorical target variable
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return self._create_exploratory_analysis(data)
        
        target_col = categorical_cols[0]  # Assume first categorical column is target
        feature_cols = numeric_cols
        
        # Prepare data
        X = data[feature_cols].dropna()
        y = data[target_col].dropna()
        
        # Ensure X and y have same indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 10 or len(y.unique()) < 2:
            return self._create_exploratory_analysis(data)
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        }
        
        best_model = None
        best_score = 0
        model_results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                model_results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if best_model is None:
            return self._create_exploratory_analysis(data)
        
        # Create visualizations
        visualizations = []
        
        # Feature importance (for Random Forest)
        if 'Random Forest' in model_results and hasattr(models['Random Forest'], 'feature_importances_'):
            rf_model = models['Random Forest']
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(importance_df, x='importance', y='feature',
                        title='Feature Importance (Random Forest)',
                        orientation='h',
                        color='importance',
                        color_continuous_scale='plasma')
            visualizations.append(fig)
        
        # Confusion matrix visualization (simplified)
        y_pred_best = model_results[type(best_model).__name__]['predictions']
        class_names = le.classes_
        
        # Create classification distribution
        pred_df = pd.DataFrame({
            'Actual': le.inverse_transform(y_test),
            'Predicted': le.inverse_transform(y_pred_best)
        })
        
        fig = px.histogram(pred_df, x='Actual', color='Predicted',
                          title='Classification Results Distribution',
                          barmode='group')
        visualizations.append(fig)
        
        results = {
            'model_performance': {
                'best_model': type(best_model).__name__,
                'accuracy': best_score,
                'target_column': target_col,
                'classes': list(class_names)
            },
            'predictions': pd.DataFrame({
                'actual': le.inverse_transform(y_test),
                'predicted': le.inverse_transform(y_pred_best)
            }),
            'visualizations': visualizations,
            'model_results': model_results
        }
        
        return results
    
    def _create_forecasting_pipeline(self, data: pd.DataFrame, 
                                    query_intent: Dict) -> Dict:
        """Create time series forecasting pipeline"""
        # For simplicity, use linear regression for trend forecasting
        
        date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Try to convert string dates if no datetime columns found
        if len(date_cols) == 0:
            for col in data.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        data[col] = pd.to_datetime(data[col])
                        date_cols.append(col)
                        break
                    except:
                        continue
        
        if len(date_cols) == 0 or len(numeric_cols) == 0:
            return self._create_regression_pipeline(data, query_intent)
        
        # Select best target column
        target_col = self._select_target_column(numeric_cols, query_intent)
        date_col = date_cols[0]
        
        forecast_data = data[[date_col, target_col]].dropna()
        forecast_data = forecast_data.sort_values(date_col)
        
        if len(forecast_data) < 5:
            return self._create_regression_pipeline(data, query_intent)
        
        # Create time features
        forecast_data['time_ordinal'] = pd.to_numeric(forecast_data[date_col])
        
        X = forecast_data[['time_ordinal']]
        y = forecast_data[target_col]
        
        # Fit linear regression for trend
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate future dates (next 30 days)
        last_date = forecast_data[date_col].max()
        try:
            future_dates = pd.date_range(last_date, periods=31, freq='D')[1:]
        except:
            # Fallback for irregular date ranges
            date_diff = (last_date - forecast_data[date_col].min()).days / len(forecast_data)
            future_dates = [last_date + pd.Timedelta(days=i*date_diff) for i in range(1, 31)]
        
        future_ordinal = pd.to_numeric(pd.Series(future_dates))
        
        # Make predictions
        future_predictions = model.predict(future_ordinal.values.reshape(-1, 1))
        
        # Create visualizations
        visualizations = []
        
        # Historical and forecasted data
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=forecast_data[date_col],
            y=forecast_data[target_col],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(
            title=f'{target_col} Forecast',
            xaxis_title='Date',
            yaxis_title=target_col,
            height=500
        )
        visualizations.append(fig)
        
        # Add trend analysis
        trend_slope = model.coef_[0] if len(model.coef_) > 0 else 0
        trend_direction = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
        
        results = {
            'model_performance': {
                'model_type': 'Linear Trend Forecast',
                'accuracy': model.score(X, y),
                'trend_direction': trend_direction,
                'trend_slope': trend_slope
            },
            'predictions': pd.DataFrame({
                'date': future_dates,
                'predicted': future_predictions
            }),
            'visualizations': visualizations,
            'forecast_period': '30 days'
        }
        
        return results
    
    def _create_clustering_pipeline(self, data: pd.DataFrame, query_intent: Dict) -> Dict:
        """Create clustering pipeline for customer segmentation"""
        from sklearn.cluster import KMeans
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return self._create_exploratory_analysis(data)
        
        # Prepare data
        X = data[numeric_cols].dropna()
        
        if len(X) < 10:
            return self._create_exploratory_analysis(data)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters (2-5)
        optimal_k = min(5, max(2, len(X) // 10))
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        clustered_data = X.copy()
        clustered_data['Cluster'] = clusters
        
        # Create visualizations
        visualizations = []
        
        # 2D scatter plot of first two numeric features
        if len(numeric_cols) >= 2:
            fig = px.scatter(clustered_data, x=numeric_cols[0], y=numeric_cols[1],
                           color='Cluster', title='Customer Clusters',
                           color_continuous_scale='viridis')
            visualizations.append(fig)
        
        # Cluster size distribution
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                    title='Cluster Size Distribution',
                    labels={'x': 'Cluster', 'y': 'Count'})
        visualizations.append(fig)
        
        results = {
            'model_performance': {
                'model_type': 'K-Means Clustering',
                'n_clusters': optimal_k,
                'accuracy': 1.0  # Clustering doesn't have traditional accuracy
            },
            'predictions': clustered_data,
            'visualizations': visualizations,
            'cluster_summary': clustered_data.groupby('Cluster')[numeric_cols].mean()
        }
        
        return results
    
    def _create_exploratory_analysis(self, data: pd.DataFrame) -> Dict:
        """Create basic exploratory analysis when ML is not applicable"""
        visualizations = []
        
        # Summary statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Distribution plots for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            fig = px.histogram(data, x=col, title=f'Distribution of {col}',
                             nbins=20, color_discrete_sequence=['skyblue'])
            visualizations.append(fig)
        
        # Bar plots for categorical columns
        for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
            if data[col].nunique() <= 10:  # Only if not too many categories
                value_counts = data[col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f'Distribution of {col}',
                           labels={'x': col, 'y': 'Count'})
                visualizations.append(fig)
        
        # Correlation heatmap if multiple numeric columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                           title='Correlation Matrix',
                           aspect='auto',
                           color_continuous_scale='RdBu')
            visualizations.append(fig)
        
        results = {
            'model_performance': {
                'analysis_type': 'Exploratory Data Analysis',
                'accuracy': 1.0,  # Always "successful" for EDA
                'data_shape': data.shape,
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols)
            },
            'summary': data.describe() if len(numeric_cols) > 0 else pd.DataFrame(),
            'visualizations': visualizations,
            'insights': self._generate_basic_insights(data)
        }
        
        return results
    
    def _generate_basic_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate basic insights from the data"""
        insights = []
        
        insights.append(f"Dataset contains {len(data)} rows and {len(data.columns)} columns")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) > 0:
            insights.append(f"Found {len(numeric_cols)} numeric columns")
            for col in numeric_cols[:3]:  # Top 3 numeric columns
                mean_val = data[col].mean()
                max_val = data[col].max()
                insights.append(f"{col}: average {mean_val:.2f}, maximum {max_val:.2f}")
        
        if len(categorical_cols) > 0:
            insights.append(f"Found {len(categorical_cols)} categorical columns")
            for col in categorical_cols[:2]:  # Top 2 categorical columns
                unique_count = data[col].nunique()
                insights.append(f"{col}: {unique_count} unique values")
        
        return insights
