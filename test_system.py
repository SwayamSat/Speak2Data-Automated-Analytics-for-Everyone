"""
Test script to verify all modules are working correctly.
Run this before launching the Streamlit app.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import config
        print("✅ config.py")
    except Exception as e:
        print(f"❌ config.py: {e}")
        return False
    
    try:
        import utils
        print("✅ utils.py")
    except Exception as e:
        print(f"❌ utils.py: {e}")
        return False
    
    try:
        import db_manager
        print("✅ db_manager.py")
    except Exception as e:
        print(f"❌ db_manager.py: {e}")
        return False
    
    try:
        import llm_client
        print("✅ llm_client.py")
    except Exception as e:
        print(f"❌ llm_client.py: {e}")
        return False
    
    try:
        import llm_task_understanding
        print("✅ llm_task_understanding.py")
    except Exception as e:
        print(f"❌ llm_task_understanding.py: {e}")
        return False
    
    try:
        import llm_sql_generator
        print("✅ llm_sql_generator.py")
    except Exception as e:
        print(f"❌ llm_sql_generator.py: {e}")
        return False
    
    try:
        import data_preprocessing
        print("✅ data_preprocessing.py")
    except Exception as e:
        print(f"❌ data_preprocessing.py: {e}")
        return False
    
    try:
        import ml_pipeline
        print("✅ ml_pipeline.py")
    except Exception as e:
        print(f"❌ ml_pipeline.py: {e}")
        return False
    
    try:
        import visualization
        print("✅ visualization.py")
    except Exception as e:
        print(f"❌ visualization.py: {e}")
        return False
    
    try:
        import experiment_logging
        print("✅ experiment_logging.py")
    except Exception as e:
        print(f"❌ experiment_logging.py: {e}")
        return False
    
    return True


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directories...")
    
    prompts_dir = Path("prompts")
    if prompts_dir.exists():
        print("✅ prompts/ directory exists")
    else:
        print("❌ prompts/ directory not found")
        return False
    
    # Check prompt files
    prompt_files = [
        "task_understanding_prompt.txt",
        "sql_generation_prompt.txt",
        "explanation_prompt.txt"
    ]
    
    for file in prompt_files:
        if (prompts_dir / file).exists():
            print(f"✅ prompts/{file}")
        else:
            print(f"❌ prompts/{file} not found")
            return False
    
    return True


def test_dependencies():
    """Test that key dependencies are installed."""
    print("\nTesting dependencies...")
    
    dependencies = [
        "streamlit",
        "pandas",
        "numpy",
        "sqlalchemy",
        "sklearn",
        "plotly",
        "google.generativeai"
    ]
    
    all_installed = True
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} not installed")
            all_installed = False
    
    return all_installed


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    try:
        from config import config
        
        print(f"✅ LLM Provider: {config.llm_provider}")
        print(f"✅ Test Size: {config.test_size}")
        print(f"✅ Random State: {config.random_state}")
        print(f"✅ Prompts Directory: {config.prompts_dir}")
        print(f"✅ Experiments DB: {config.experiments_db}")
        
        return True
    
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Speak2Data System Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Directory Test", test_directories),
        ("Dependency Test", test_dependencies),
        ("Configuration Test", test_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"{test_name}")
        print('=' * 60)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All tests passed! System is ready.")
        print("\nTo run the application:")
        print("  streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues before running the app.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
