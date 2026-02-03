"""
Setup Helper - Downloads sample images for testing the face detection demo
This script helps you quickly get started with sample images
"""

import os
import requests
from pathlib import Path

def download_sample_image(url, filename, folder):
    """Download an image from URL"""
    filepath = os.path.join(folder, filename)
    
    if os.path.exists(filepath):
        print(f"  ⏭️  {filename} already exists, skipping...")
        return
    
    try:
        print(f"  ⬇️  Downloading {filename}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"  ✅ Downloaded {filename}")
    except Exception as e:
        print(f"  ❌ Failed to download {filename}: {str(e)}")

def setup_sample_images():
    """
    Note: For a real demo, you should add your own images.
    This is just a placeholder to show the structure.
    """
    print("\n" + "="*60)
    print("Sample Images Setup Helper")
    print("="*60)
    
    print("\n📁 Creating directories...")
    os.makedirs('sample_faces', exist_ok=True)
    os.makedirs('test_images', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    print("  ✅ Directories created")
    
    print("\n" + "="*60)
    print("IMPORTANT: Add Your Own Images")
    print("="*60)
    print("\nTo use this demo, you need to add:")
    print("\n1️⃣  Reference faces to 'sample_faces/' folder:")
    print("   - Add clear photos of people you want to recognize")
    print("   - Name files: PersonName.jpg (e.g., John_Doe.jpg)")
    print("   - Use well-lit, front-facing photos")
    
    print("\n2️⃣  Test images to 'test_images/' folder:")
    print("   - Add any images with faces to detect")
    print("   - Can be group photos or individual portraits")
    print("   - Supports: JPG, PNG, BMP formats")
    
    print("\n3️⃣  Then run:")
    print("   python face_detection_demo.py")
    
    print("\n" + "="*60)
    print("Quick Start:")
    print("="*60)
    print("1. Add images to sample_faces/ and test_images/")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run demo: python face_detection_demo.py")
    print("4. Check results in output/ folder")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    setup_sample_images()
