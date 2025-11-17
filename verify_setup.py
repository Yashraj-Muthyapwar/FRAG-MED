#!/usr/bin/env python3
"""
Verify that all local models and data are properly configured
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
import json

def verify_setup():
    print("="*80)
    print("FRAG-MED SETUP VERIFICATION")
    print("="*80)
    
    checks = []
    
    # Check 1: Embedding model
    embedding_path = config.EMBEDDING_MODEL_PATH
    print(f"\n📦 Embedding Model:")
    print(f"   Path/Name: {embedding_path}")
    
    if Path(embedding_path).exists():
        # Local path
        model_files = list(Path(embedding_path).rglob('*.bin')) + \
                     list(Path(embedding_path).rglob('*.safetensors'))
        
        if model_files:
            print(f"✅ Local model found with {len(model_files)} weight file(s)")
            checks.append(True)
        else:
            print(f"⚠️  Local model directory exists but missing weight files")
            print(f"   Will fallback to HuggingFace auto-download")
            checks.append(True)
    else:
        # HuggingFace name
        print(f"ℹ️  Will use HuggingFace auto-download")
        print(f"   Model will be downloaded on first use (~420MB)")
        checks.append(True)
    
    # Check 2: Raw data
    patient_files = list(config.RAW_DATA_DIR.glob("*.json"))
    if patient_files:
        print(f"\n✅ Found {len(patient_files)} patient files")
        
        # Check sample file
        with open(patient_files[0], 'r') as f:
            sample = json.load(f)
            has_devices = 'patient_level_devices' in sample
            print(f"{'✅' if has_devices else '⚠️ '} Patient devices field: {has_devices}")
            checks.append(True)
    else:
        print(f"\n❌ No patient files found in {config.RAW_DATA_DIR}")
        checks.append(False)
    
    # Check 3: Ollama
    print(f"\n🧠 LLM Model:")
    import subprocess
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'bio-mistral' in result.stdout:
            print(f"✅ Ollama model available: {config.LLM_MODEL_NAME}")
            checks.append(True)
        else:
            print(f"❌ Ollama model not found: {config.LLM_MODEL_NAME}")
            print(f"   Run: ollama pull {config.LLM_MODEL_NAME}")
            checks.append(False)
    except FileNotFoundError:
        print("❌ Ollama not installed")
        checks.append(False)
    
    print("\n" + "="*80)
    if all(checks):
        print("🎉 READY TO RUN!")
        print("\nNext step:")
        print("  python src/main_preprocessing.py")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nTo fix:")
        print("  1. Run: python download_local_models.py (for embedding model)")
        print("  2. Or just run preprocessing (will auto-download)")
    print("="*80)
    
    return all(checks)

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
