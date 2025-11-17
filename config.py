"""
Configuration management for the Speak2Data system.
Handles API keys, database connections, model defaults, and paths.
"""
import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Main configuration class for the application."""
    
    def __init__(self):
        # LLM Configuration
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini")  # "gemini", "openai", or "anthropic"
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        # LLM Model Names
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        
        # Database Configuration
        self.database_url = os.getenv("DATABASE_URL", "")
        self.max_query_rows = int(os.getenv("MAX_QUERY_ROWS", "10000"))
        self.sample_threshold = int(os.getenv("SAMPLE_THRESHOLD", "50000"))
        
        # ML Configuration
        self.test_size = float(os.getenv("TEST_SIZE", "0.2"))
        self.random_state = int(os.getenv("RANDOM_STATE", "42"))
        self.cv_folds = int(os.getenv("CV_FOLDS", "5"))
        
        # Classification Models
        self.classification_models = [
            "LogisticRegression",
            "RandomForestClassifier",
            "GradientBoostingClassifier"
        ]
        
        # Regression Models
        self.regression_models = [
            "LinearRegression",
            "RandomForestRegressor",
            "GradientBoostingRegressor"
        ]
        
        # Clustering Models
        self.clustering_models = ["KMeans", "DBSCAN"]
        self.kmeans_k_range = (2, 6)  # Min and max K to try
        
        # Paths
        self.project_root = Path(__file__).parent
        self.prompts_dir = self.project_root / "prompts"
        self.experiments_db = self.project_root / "experiments.db"
        self.temp_db_dir = self.project_root / "temp_dbs"
        
        # Create directories if they don't exist
        self.prompts_dir.mkdir(exist_ok=True)
        self.temp_db_dir.mkdir(exist_ok=True)
        
        # Preprocessing defaults
        self.missing_strategy = "drop"  # "drop", "mean", "median", "mode"
        self.categorical_encoding = "onehot"  # "onehot", "ordinal"
        self.scaling_method = "standard"  # "standard", "minmax", "robust", "none"
        
        # Visualization
        self.plot_style = "plotly"
        self.color_scheme = "viridis"
    
    def get_llm_api_key(self) -> str:
        """Get the API key for the current LLM provider."""
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        elif self.llm_provider == "openai":
            return self.openai_api_key
        elif self.llm_provider == "anthropic":
            return self.anthropic_api_key
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def get_llm_model_name(self) -> str:
        """Get the model name for the current LLM provider."""
        if self.llm_provider == "gemini":
            return self.gemini_model
        elif self.llm_provider == "openai":
            return self.openai_model
        elif self.llm_provider == "anthropic":
            return self.anthropic_model
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (for logging)."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.get_llm_model_name(),
            "test_size": self.test_size,
            "random_state": self.random_state,
            "missing_strategy": self.missing_strategy,
            "categorical_encoding": self.categorical_encoding,
            "scaling_method": self.scaling_method
        }


# Global config instance
config = Config()
