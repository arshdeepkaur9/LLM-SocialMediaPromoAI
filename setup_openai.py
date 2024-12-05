import os
import subprocess
import sys

def install_dependencies():
    """
    Installs required dependencies for OpenAI functionality.
    """
    print("Installing required packages...")
    packages = ["openai==1.56.2", "streamlit", "pandas", "langchain", "tqdm", "httpx", "anyio"]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
        print("All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        sys.exit(1)

def configure_openai_api_key():
    """
    Prompts the user to add the OpenAI API key to environment variables.
    """
    api_key = input("Enter your OpenAI API Key: ").strip()
    if not api_key:
        print("API Key is required!")
        sys.exit(1)
    
    # Add API key to environment variables
    os.environ["OPENAI_API_KEY"] = api_key
    print("API Key configured successfully!")

def check_openai_cli():
    """
    Verifies the OpenAI CLI is installed and accessible.
    """
    print("Checking OpenAI CLI installation...")
    try:
        result = subprocess.run(["openai", "--help"], check=True, capture_output=True, text=True)
        print("OpenAI CLI is installed and working!")
    except FileNotFoundError:
        print("OpenAI CLI is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
        print("OpenAI CLI installed successfully!")

def migrate_openai_code():
    """
    Runs the OpenAI migrate command to update code to the latest API.
    """
    print("Running OpenAI migration...")
    try:
        subprocess.check_call(["openai", "migrate", "."])
        print("Code migrated successfully!")
    except Exception as e:
        print(f"Error during migration: {e}")

if __name__ == "__main__":
    install_dependencies()
    configure_openai_api_key()
    check_openai_cli()
    migrate_openai_code()