#!/usr/bin/env python3
"""
Verify training file format for Together.ai
"""

from together import Together
import os
import json

# Set API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    print("❌ Please set TOGETHER_API_KEY environment variable")
    exit(1)

print("🔍 Verifying training file format...")

# Initialize client
client = Together(api_key=TOGETHER_API_KEY)

# Check our training file
try:
    from together.utils import check_file
    
    # Check the file format
    sft_report = check_file("/home/nuko/dev/DeepDiscord/together_fine_tuning/data/sherif_together_uncensored.jsonl")
    
    print("\n📊 File Format Check Results:")
    print(json.dumps(sft_report, indent=2))
    
    if sft_report.get("is_check_passed") == True:
        print("\n✅ File format is valid!")
        
        # Upload the file
        print("\n📤 Uploading file to Together.ai...")
        train_file_resp = client.files.upload(
            "/home/nuko/dev/DeepDiscord/together_fine_tuning/data/sherif_together_uncensored.jsonl", 
            check=True
        )
        
        print(f"\n✅ File uploaded successfully!")
        print(f"📁 File ID: {train_file_resp.id}")
        print(f"\n🚀 Use this ID for fine-tuning:")
        print(f"together fine-tuning create \\")
        print(f'  --model "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference" \\')
        print(f'  --training-file "{train_file_resp.id}" \\')
        print(f'  --suffix "sherif-chaos-8b" \\')
        print(f"  --n-epochs 3")
        
    else:
        print("\n❌ File format check failed!")
        print("Issues found:")
        if "errors" in sft_report:
            for error in sft_report["errors"]:
                print(f"  - {error}")
                
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()