import os
import requests
import json
import time
from dotenv import load_dotenv
from typing import Dict, Optional

class OpenRouterGeminiTest:
    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.results = {
            "env_file_loaded": False,
            "api_key_present": False,
            "api_key_valid": False,
            "connection_successful": False,
            "model_response_received": False
        }

    def load_environment(self) -> bool:
        """Load environment variables from .env file"""
        try:
            load_dotenv()
            self.results["env_file_loaded"] = True
            print("✅ Environment file loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load .env file: {e}")
            return False

    def check_api_key(self) -> bool:
        """Check if OpenRouter API key is present"""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if self.api_key:
            self.results["api_key_present"] = True
            print(f"✅ API Key found (length: {len(self.api_key)})")
            print(f"🔑 API Key preview: {self.api_key[:5]}...{self.api_key[-5:]}")
            return True
        else:
            print("❌ OPENROUTER_API_KEY not found in environment variables")
            return False

    def test_api_connection(self) -> Dict[str, any]:
        """Test OpenRouter API connection with Gemini Pro model"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",  # Replace with your site URL if needed
            "X-Title": "API Test"  # Replace with your site name if needed
        }

        data = {
            "model": "google/gemini-2.5-pro",  # Using Gemini Pro through OpenRouter
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Can you confirm you're Gemini Pro through OpenRouter?"
                }
            ]
        }

        try:
            start_time = time.time()
            print("\n🔄 Testing OpenRouter API connection with Gemini Pro...")
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=30
            )

            end_time = time.time()
            response_time = end_time - start_time

            if response.status_code == 200:
                self.results["connection_successful"] = True
                self.results["model_response_received"] = True
                response_data = response.json()
                
                print(f"✅ Connection successful (Response time: {response_time:.2f}s)")
                print("\n📝 Response details:")
                print(f"Model: {response_data.get('model', 'N/A')}")
                print(f"Response: {response_data.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
                print(f"\nUsage statistics:")
                print(json.dumps(response_data.get('usage', {}), indent=2))
                
                return response_data
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Error details: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Connection error: {str(e)}")
            return None

def run_tests():
    """Run all OpenRouter Gemini Pro API tests"""
    print("🚀 Starting OpenRouter Gemini Pro API tests...\n")
    
    tester = OpenRouterGeminiTest()
    
    # Run tests in sequence
    if not tester.load_environment():
        return tester.results
    
    if not tester.check_api_key():
        return tester.results
    
    response_data = tester.test_api_connection()
    
    print("\n📊 Test Results Summary:")
    for key, value in tester.results.items():
        status = "✅" if value else "❌"
        print(f"{status} {key.replace('_', ' ').title()}")

    return tester.results

if __name__ == "__main__":
    run_tests()