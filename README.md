# Real-Time Emotion Detection Application

This application provides real-time emotion detection from webcam video using a CNN model trained on facial expressions.

## Requirements

- Python 3.9+
- TensorFlow 2.x
- OpenCV 4.x
- Flask
- Flask-CORS
- Bun.js 1.x

0. **Grab the Model Copy**

go to this link: https://drive.google.com/drive/folders/1uAWoaOlrq9R5DS7HNPUhn728iTV2azZu?usp=sharing

and download the model and store it under new folder `best_models` at your root directory

## Setup

1. **Install Python dependencies**

```bash
pip install tensorflow opencv-python flask flask-cors
```

2. **Install Bun.js** (if not already installed)

```bash
# For macOS/Linux
curl -fsSL https://bun.sh/install | bash

# For Windows via WSL
curl -fsSL https://bun.sh/install | bash

# Verify installation
bun --version
```

3. **Install Bun.js dependencies**

```bash
bun install
```

## Running the Application

1. **Start the Flask server**

```bash
# From the root directory
python app.py --port=3440
```

2. **Start the Bun.js server**

```bash
# From the root directory
bun run server.js
```

3. **Access the application**

Open your browser and navigate to:
```
http://localhost:3000
```

4. **Using the application**

- Click "Start Camera" to enable webcam access
- Click "Start Capture" to begin emotion detection
- Your facial emotion will be detected and displayed in real-time

## Troubleshooting

### Common Issues

1. **"Address already in use" error**
   - Another process is using the port. Change the port in the command: `python app.py --port=3441`

2. **Camera not detected**
   - Ensure your browser has permission to access the camera
   - Try a different browser

3. **Model loading error**
   - Ensure the model file (`model_cnn_1.keras`) is in the correct location (best_model/`)

4. **Image decoding errors**
   - Check the server logs for specific error messages
   - Try the test endpoint: `http://localhost:3000/api/test-image`

## API Endpoints

- `/predict-simple`: Processes an image and returns emotion predictions
- `/health`: Status check endpoint
- `/test`: Tests the model with a sample image
- `/api/test-image`: Tests the full pipeline with a sample image

## Debugging

- Debug files are saved in the server directory:
  - `debug_decoded.jpg`: Successfully decoded image
  - `debug_from_base64.bin`: Raw binary data from base64 decoding
  - `debug_request_blob.bin`: Raw blob data from client

## Project Structure

- `app.py`: Flask server for model inference
- `server.js`: Bun.js server for client/server communication
- `index.html`: Web interface for real-time emotion detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.
