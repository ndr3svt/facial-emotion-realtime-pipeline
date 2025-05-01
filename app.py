# app.py - Flask server for emotion detection

import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS
import traceback

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configuration
MODEL_PATH = '../best_model/model_cnn_1.keras'
IMG_SIZE = 48
CLASS_NAMES = ['happy', 'neutral', 'sad', 'surprise']  # Update with your actual class names

# Load the model at startup
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Function to preprocess a single face image
def preprocess_face(face):
    """Preprocess a face image for prediction."""
    # Resize to match model input size
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    
    # Ensure grayscale (our model expects 1 channel)
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values to [0, 1]
    face = face.astype('float32') / 255.0
    
    # Reshape to match model input shape (batch_size, height, width, channels)
    face = np.expand_dims(face, axis=-1)  # Add channel dimension
    face = np.expand_dims(face, axis=0)   # Add batch dimension
    
    return face

# Face detection function
def detect_face(image_data):
    """Detect faces in the image and return the largest face."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if image was decoded successfully
        if img is None:
            print("Failed to decode image data")
            return None
        
        print(f"Image shape: {img.shape}")
        
        # Load pre-trained face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Check if the cascade file exists
        if not os.path.exists(cascade_path):
            print(f"Cascade file not found at: {cascade_path}")
            # Try an alternative path
            cascade_path = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                print("Alternative path also not found")
                # Final fallback - try relative path
                cascade_path = 'haarcascade_frontalface_default.xml'
                if not os.path.exists(cascade_path):
                    raise FileNotFoundError("Could not find face cascade file")
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Check if cascade loaded properly
        if face_cascade.empty():
            raise ValueError("Failed to load face cascade classifier")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # If no faces detected, return None
        if len(faces) == 0:
            print("No faces detected in the image")
            return None
        
        # Get the largest face (assuming it's the main subject)
        largest_face = None
        largest_area = 0
        face_location = None
        
        for (x, y, w, h) in faces:
            if w * h > largest_area:
                largest_area = w * h
                largest_face = gray[y:y+h, x:x+w]
                face_location = (x, y, w, h)
        
        return largest_face, face_location
    
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        traceback.print_exc()
        return None

# Simplified prediction function that doesn't use face detection
def predict_emotion_direct(image):
    """Make a prediction directly from an image without face detection."""
    # Resize to expected size
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Make sure it's grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add dimensions
    img = np.expand_dims(img, axis=-1)  # Channel
    img = np.expand_dims(img, axis=0)   # Batch
    
    # Predict
    prediction = model.predict(img, verbose=0)[0]
    
    # Get results
    predicted_class_index = np.argmax(prediction)
    confidence = float(prediction[predicted_class_index])
    emotion = CLASS_NAMES[predicted_class_index]
    
    return {
        'emotion': emotion,
        'confidence': confidence,
        'all_scores': {CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))}
    }

# API endpoint for predictions with face detection
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            print("No image field in request")
            return jsonify({'error': 'No image provided'}), 400
        
        # Get the image from the request
        file = request.files['image']
        
        # Check if the file has content
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'Empty file'}), 400
        
        # Read the file data
        image_data = file.read()
        
        print(f"Received image data size: {len(image_data)} bytes")
        
        if len(image_data) == 0:
            print("Empty image data")
            return jsonify({'error': 'Empty image data'}), 400
        
        # Save the received image for debugging
        with open('debug_received.jpg', 'wb') as f:
            f.write(image_data)
        
        # Detect face in the image
        face_result = detect_face(image_data)
        
        if face_result is None:
            # Fallback to direct image processing without face detection
            print("Falling back to direct image processing")
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                return jsonify({'error': 'Failed to decode image'}), 400
            
            result = predict_emotion_direct(img)
            # Add placeholder face location for compatibility
            result['face_location'] = {'x': 0, 'y': 0, 'width': img.shape[1], 'height': img.shape[0]}
            return jsonify(result)
        
        face, (x, y, w, h) = face_result
        
        # Preprocess the face
        processed_face = preprocess_face(face)
        
        # Make a prediction
        prediction = model.predict(processed_face, verbose=0)[0]
        
        # Get the predicted class and confidence
        predicted_class_index = np.argmax(prediction)
        confidence = float(prediction[predicted_class_index])
        emotion = CLASS_NAMES[predicted_class_index]
        
        # Return the prediction results
        result = {
            'emotion': emotion,
            'confidence': confidence,
            'face_location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'all_scores': {CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))}
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Simplified endpoint that skips face detection
@app.route('/predict-simple', methods=['POST'])
def predict_simple():
    """A simplified prediction endpoint that directly processes images without face detection."""
    try:
        print("Received request to /predict-simple")
        image_data = None
        
        # 1. Handle JSON requests with base64 data
        if request.is_json:
            print("Processing JSON request")
            data = request.json
            
            if 'image_base64' not in data:
                return jsonify({'error': 'Missing image_base64 field in JSON data'}), 400
            
            # Get and clean the base64 string
            base64_string = data['image_base64'].strip()
            print(f"Received base64 data of length: {len(base64_string)}")
            
            # Save a sample for debugging
            with open('debug_base64_sample.txt', 'w') as f:
                f.write(base64_string[:100] + '...')
            
            # Ensure proper padding
            padding = 4 - (len(base64_string) % 4)
            if padding < 4:
                base64_string += '=' * padding
                print(f"Added {padding} padding characters")
            
            # Decode base64 to binary
            try:
                import base64
                image_data = base64.b64decode(base64_string)
                print(f"Successfully decoded base64 data: {len(image_data)} bytes")
                
                # Save the binary data for debugging
                with open('debug_from_base64.bin', 'wb') as f:
                    f.write(image_data)
            except Exception as e:
                print(f"Base64 decoding error: {str(e)}")
                return jsonify({'error': f'Failed to decode base64 data: {str(e)}'}), 400
        
        # 2. Handle form data with image file
        elif 'image' in request.files:
            print("Processing file upload")
            file = request.files['image']
            image_data = file.read()
            print(f"Read {len(image_data)} bytes from uploaded file")
            
            # Save the uploaded file for debugging
            with open('debug_uploaded_file.bin', 'wb') as f:
                f.write(image_data)
        
        # 3. No valid data found
        else:
            return jsonify({'error': 'No image data found in request'}), 400
        
        # 4. Process the image data
        print("Processing image data...")
        try:
            # Convert binary data to numpy array
            np_arr = np.frombuffer(image_data, np.uint8)
            
            # Try different decoding methods
            # Method 1: Try grayscale decoding
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print(f"Successfully decoded as grayscale image: {img.shape}")
            else:
                # Method 2: Try color decoding then convert to grayscale
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    print(f"Successfully decoded as color image: {img.shape}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    # Failed to decode
                    return jsonify({'error': 'Could not decode image data'}), 400
            
            # Save the decoded image for verification
            cv2.imwrite('debug_decoded.jpg', img)
            
            # 5. Make prediction
            print("Making prediction...")
            # Resize to expected input size
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Normalize
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add channel dimension for model input
            img_input = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
            img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension
            
            # Predict
            prediction = model.predict(img_input, verbose=0)[0]
            
            # Process results
            predicted_class_index = np.argmax(prediction)
            confidence = float(prediction[predicted_class_index])
            emotion = CLASS_NAMES[predicted_class_index]
            
            # 6. Return prediction
            result = {
                'emotion': emotion,
                'confidence': confidence,
                'face_location': {'x': 0, 'y': 0, 'width': img.shape[1], 'height': img.shape[0]},
                'all_scores': {CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))}
            }
            
            print(f"Prediction successful: {emotion} ({confidence:.2f})")
            return jsonify(result)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Image processing error: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Basic test endpoint
@app.route('/test', methods=['GET'])
def test():
    try:
        # Load a test image directly from file
        test_image_path = '../Facial_emotion_images/test/surprise/155.jpg'
        
        if not os.path.exists(test_image_path):
            # Try to find any image in the dataset directory for testing
            for root, dirs, files in os.walk('../Facial_emotion_images'):
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png'):
                        test_image_path = os.path.join(root, file)
                        break
                if test_image_path != '../Facial_emotion_images/test/surprise/155.jpg':
                    break
        
        if not os.path.exists(test_image_path):
            return jsonify({'error': 'No test images found in dataset directory'}), 404
        
        print(f"Using test image: {test_image_path}")
        
        with open(test_image_path, 'rb') as f:
            image_data = f.read()
        
        # Process the image with face detection
        face_result = detect_face(image_data)
        
        if face_result is None:
            # Fallback to direct processing
            img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
            result = predict_emotion_direct(img)
            result['face_location'] = {'x': 0, 'y': 0, 'width': img.shape[1], 'height': img.shape[0]}
            result['note'] = 'Used direct image processing (no face detected)'
            return jsonify(result)
        
        face, (x, y, w, h) = face_result
        
        # Preprocess the face
        processed_face = preprocess_face(face)
        
        # Make a prediction
        prediction = model.predict(processed_face, verbose=0)[0]
        
        # Get the predicted class and confidence
        predicted_class_index = np.argmax(prediction)
        confidence = float(prediction[predicted_class_index])
        emotion = CLASS_NAMES[predicted_class_index]
        
        # Return the prediction results
        result = {
            'emotion': emotion,
            'confidence': confidence,
            'face_location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'all_scores': {CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))},
            'test_image': test_image_path
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Test endpoint error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Test error: {str(e)}'}), 500

# Simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok', 
        'model': 'loaded',
        'class_names': CLASS_NAMES
    })

@app.route('/debug-image', methods=['POST'])
def debug_image():
    """Debug endpoint to visualize how images are preprocessed before going to the model."""
    try:
        image_data = None
        
        # Handle JSON request with base64 data
        if request.is_json:
            data = request.json
            if 'image_base64' not in data:
                return jsonify({'error': 'No image_base64 field in request'}), 400
                
            # Decode base64
            import base64
            base64_string = data['image_base64'].strip()
            
            # Ensure proper padding
            padding = 4 - (len(base64_string) % 4)
            if padding < 4:
                base64_string += '=' * padding
                
            try:
                image_data = base64.b64decode(base64_string)
            except Exception as e:
                return jsonify({'error': f'Failed to decode base64: {str(e)}'}), 400
                
        # Handle multipart/form-data
        elif 'image' in request.files:
            file = request.files['image']
            image_data = file.read()
        else:
            return jsonify({'error': 'No image data found in request'}), 400
            
        # Decode image
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
            
        # Create debug info
        debug_info = {}
        
        # Original dimensions
        debug_info['original_shape'] = img.shape
        debug_info['original_dtype'] = str(img.dtype)
        debug_info['original_min'] = float(np.min(img))
        debug_info['original_max'] = float(np.max(img))
        debug_info['original_mean'] = float(np.mean(img))
        
        # Save original
        cv2.imwrite('debug_original.jpg', img)
        
        # Process like we do for the model
        # Resize to 48x48
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        debug_info['resized_shape'] = img_resized.shape
        
        # Convert to float and normalize
        img_normalized = img_resized.astype('float32') / 255.0
        debug_info['normalized_min'] = float(np.min(img_normalized))
        debug_info['normalized_max'] = float(np.max(img_normalized))
        debug_info['normalized_mean'] = float(np.mean(img_normalized))
        
        # Save intermediate images as grayscale JPG for visualization
        cv2.imwrite('debug_resized.jpg', img_resized)
        
        # Create a visualization of normalized image (scale back to 0-255 for saving)
        normalized_viz = (img_normalized * 255).astype(np.uint8)
        cv2.imwrite('debug_normalized.jpg', normalized_viz)
        
        # Prepare model input (with channel and batch dimensions)
        img_input = np.expand_dims(img_normalized, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)
        debug_info['model_input_shape'] = list(img_input.shape)
        
        # Run a prediction to verify
        prediction = model.predict(img_input, verbose=0)[0]
        predicted_class_index = np.argmax(prediction)
        confidence = float(prediction[predicted_class_index])
        emotion = CLASS_NAMES[predicted_class_index]
        
        debug_info['predicted_emotion'] = emotion
        debug_info['confidence'] = confidence
        debug_info['all_scores'] = {CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))}
        
        # Return the debug info
        return jsonify({
            'message': 'Image processing debug information',
            'debug_info': debug_info,
            'debug_images': {
                'original': '/debug_original.jpg',
                'resized': '/debug_resized.jpg',
                'normalized': '/debug_normalized.jpg'
            }
        })
        
    except Exception as e:
        print(f"Error in debug endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Debug error: {str(e)}'}), 500

# Add a static route to serve debug images
@app.route('/<path:filename>')
def serve_debug_image(filename):
    if filename.startswith('debug_') and filename.endswith('.jpg'):
        return send_file(filename, mimetype='image/jpeg')
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Emotion Detection Server')
    parser.add_argument('--port', type=int, default=3440, help='Port to run the server on')
    args = parser.parse_args()
    
    # Run the Flask app, allowing connections from any IP
    app.run(host='0.0.0.0', port=args.port, debug=False)