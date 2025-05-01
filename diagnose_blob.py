#!/usr/bin/env python3
"""
Diagnostic script to examine the binary blob saved from the client.
This helps identify issues with image encoding/decoding.
"""

import os
import sys
import binascii
import cv2
import numpy as np
from PIL import Image
import io

def examine_blob(blob_path):
    """Examine a binary blob and try to diagnose image issues."""
    
    if not os.path.exists(blob_path):
        print(f"Error: File not found: {blob_path}")
        return False
    
    # Get file size
    file_size = os.path.getsize(blob_path)
    print(f"File size: {file_size} bytes")
    
    # Read the binary data
    with open(blob_path, 'rb') as f:
        data = f.read()
    
    # Check first few bytes to identify file type
    print(f"First 20 bytes: {binascii.hexlify(data[:20])}")
    
    # Check for common image headers
    file_type = "Unknown"
    if data.startswith(b'\xff\xd8\xff'):
        file_type = "JPEG"
    elif data.startswith(b'\x89PNG\r\n\x1a\n'):
        file_type = "PNG"
    elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
        file_type = "GIF"
    elif data.startswith(b'BM'):
        file_type = "BMP"
    elif data.startswith(b'data:image/'):
        file_type = "Data URL"
        # Try to extract the base64 data
        try:
            data_url = data.decode('utf-8', errors='ignore')
            if ';base64,' in data_url:
                print("Found data URL with base64 encoding")
                base64_prefix = data_url.find(';base64,') + 8
                base64_data = data_url[base64_prefix:]
                print(f"Base64 data length: {len(base64_data)}")
        except Exception as e:
            print(f"Error decoding data URL: {str(e)}")
    
    print(f"Detected file type: {file_type}")
    
    # Try to decode as image using different methods
    success = False
    
    # Method 1: OpenCV
    try:
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            print(f"Successfully decoded with OpenCV: {img.shape}")
            cv2.imwrite('debug_opencv.jpg', img)
            success = True
        else:
            print("Failed to decode with OpenCV")
    except Exception as e:
        print(f"OpenCV error: {str(e)}")
    
    # Method 2: PIL
    try:
        pil_img = Image.open(io.BytesIO(data))
        print(f"Successfully decoded with PIL: {pil_img.format}, {pil_img.size}, {pil_img.mode}")
        pil_img.save('debug_pil.jpg')
        success = True
    except Exception as e:
        print(f"PIL error: {str(e)}")
    
    # If it looks like a text file, try to read it as such
    if not success and file_type == "Unknown":
        try:
            text_data = data.decode('utf-8', errors='ignore')
            print(f"First 100 chars as text: {text_data[:100]}")
            
            # Check if it's base64 encoded data
            if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in text_data.strip()):
                print("Data appears to be base64 encoded")
                
                # Try to decode the base64 data
                import base64
                try:
                    decoded = base64.b64decode(text_data.strip())
                    print(f"Decoded base64 data length: {len(decoded)}")
                    
                    # Write decoded data to file
                    with open('debug_decoded_base64.bin', 'wb') as f:
                        f.write(decoded)
                    
                    # Try to decode the result as an image
                    try:
                        nparr = np.frombuffer(decoded, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            print(f"Successfully decoded base64 data as image: {img.shape}")
                            cv2.imwrite('debug_base64_opencv.jpg', img)
                            success = True
                    except Exception as e:
                        print(f"Failed to decode base64 data as image: {str(e)}")
                        
                except Exception as e:
                    print(f"Failed to decode as base64: {str(e)}")
        except Exception as e:
            print(f"Failed to decode as text: {str(e)}")
    
    return success

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_blob.py <blob_file_path>")
        sys.exit(1)
        
    blob_path = sys.argv[1]
    success = examine_blob(blob_path)
    print(f"Diagnostic complete. Success: {success}")
    sys.exit(0 if success else 1) 