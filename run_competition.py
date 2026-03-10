"""
Quick runner for competition submission
Run this file to start the app
"""

import os
import sys
import subprocess

def main():
    print("="*60)
    print("🏆 COMPETITION-GRADE FAKE NEWS DETECTOR")
    print("="*60)
    print("\nChecking requirements...")
    
    # Check if model needs training
    if not os.path.exists('competition_model.pkl'):
        print("\n📊 First-time setup: Training model...")
        print("This will take about 30 seconds...")
        
        from competition_model import CompetitionFakeNewsDetector, create_training_data
        detector = CompetitionFakeNewsDetector()
        texts, labels = create_training_data()
        metrics = detector.train(texts, labels)
        detector.save()
        
        print(f"\n✅ Model trained successfully!")
        print(f"   Accuracy: {metrics['accuracy']:.1%}")
        print(f"   F1-Score: {metrics['f1']:.1%}")
    else:
        print("✅ Model already trained")
    
    print("\n🚀 Starting Streamlit app...")
    print("📱 Opening in browser...\n")
    
    # Run streamlit
    subprocess.run(["streamlit", "run", "competition_app.py"])

if __name__ == "__main__":
    main()