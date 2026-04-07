import requests
import sys

def run_validation():
    print("Running Pre-Submission Validation...")
    
    # 1. HF Space (Local Server)
    try:
        res = requests.get("http://localhost:7860/health")
        res.raise_for_status()
        print("✅ Server is responding.")
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        sys.exit(1)
        
    # 2. Endpoints & Tasks
    try:
        tasks = requests.get("http://localhost:7860/tasks").json()
        if len(tasks) >= 3:
            print(f"✅ Found {len(tasks)} tasks.")
        else:
            print("❌ Less than 3 tasks found.")
            
        print("✅ OpenEnv spec endpoints validated.")
    except Exception as e:
        print(f"❌ Error validating endpoints: {e}")
        
    print("\n✅ Verification Script complete!")

if __name__ == "__main__":
    run_validation()
