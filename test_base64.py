#!/usr/bin/env python3
"""
Test script to verify Flask API works with base64 encoded images.
This script takes an image file, converts it to base64, and sends it to the Flask server.
"""

import requests
import base64
import json
import argparse
import os
import sys

def test_base64_api(image_path, endpoint='predict-simple', port=3440):
    """Test the Flask API with a base64 encoded image."""
    
    url = f'http://localhost:{port}/{endpoint}'
    print(f"Testing endpoint: {url}")
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    # Get file size
    file_size = os.path.getsize(image_path)
    print(f"Image file size: {file_size} bytes")
    
    # Read the image file and convert to base64
    with open(image_path, 'rb') as img_file:
        image_data = img_file.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')
    
    print(f"Base64 encoded size: {len(base64_data)} characters")
    
    # Create JSON payload with base64 image
    payload = {
        'image_base64': base64_data
    }
    
    # Send request
    print("Sending request...")
    response = requests.post(
        url, 
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Response status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Successful response:")
        print(f"Detected emotion: {result.get('emotion')}")
        print(f"Confidence: {result.get('confidence')}")
        print("All scores:", result.get('all_scores'))
        return True
    else:
        print(f"Error response ({response.status_code}):")
        print(response.text)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Flask Emotion API with base64 encoded image")
    parser.add_argument("image_path", help="Path to the image file to test")
    parser.add_argument("--port", type=int, default=3440, help="Flask server port")
    parser.add_argument("--endpoint", default="predict-simple", 
                        choices=["predict", "predict-simple"], 
                        help="Endpoint to test")
    
    args = parser.parse_args()
    
    success = test_base64_api(args.image_path, args.endpoint, args.port)
    sys.exit(0 if success else 1) 