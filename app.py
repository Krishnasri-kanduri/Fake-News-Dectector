"""
Simple runner for Fake News Detection System
"""

import os
import subprocess

def main():
    print("="*60)
    print("📰 FAKE NEWS DETECTION SYSTEM")
    print("="*60)
    
    print("\n🚀 Starting application...")
    print("📱 Opening in browser...\n")
    
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()