"""
Setup script for Speak2Data application.
Helps users get started quickly with the application.
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file from template."""
    print("\n🔧 Setting up environment file...")
    
    if os.path.exists(".env"):
        print("✅ .env file already exists")
        return True
    
    if os.path.exists("env_template.txt"):
        shutil.copy("env_template.txt", ".env")
        print("✅ .env file created from template")
        print("⚠️  Please edit .env file and add your GEMINI_API_KEY")
        return True
    else:
        print("❌ env_template.txt not found")
        return False

def test_imports():
    """Test if all modules can be imported."""
    print("\n🧪 Testing imports...")
    
    modules_to_test = [
        "streamlit",
        "pandas", 
        "numpy",
        "sklearn",
        "plotly",
        "sqlalchemy"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful")
    return True

def run_basic_test():
    """Run basic functionality test."""
    print("\n🧪 Running basic tests...")
    try:
        result = subprocess.run([sys.executable, "test_app.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Basic tests passed")
            return True
        else:
            print(f"❌ Tests failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Could not run tests: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Speak2Data Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create .env file
    if not create_env_file():
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    # Run basic test
    if not run_basic_test():
        print("⚠️  Basic tests failed, but setup may still work")
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your GEMINI_API_KEY")
    print("2. Run: streamlit run app.py")
    print("3. Open your browser to http://localhost:8501")
    print("\nFor more information, see README.md")
    
    return True

if __name__ == "__main__":
    main()
