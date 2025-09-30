# SPaCial OCR Client
<img width="1495" height="735" alt="image" src="https://github.com/user-attachments/assets/138f8c9b-3ec6-402a-8493-b775866bc112" />

A client application to interact with the SPaCial AI OCR Service hosted on Hugging Face Spaces.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/spacial-ocr-client)

## Features

- 📝 Regular text detection (green bounding boxes)
- 📐 Dimension detection (blue bounding boxes) 
- ⚡ Fast and accurate OCR modes
- 🔄 Image rotation support
- 🖼️ Interactive web interface with Gradio

## 🚀 2-Step Deploy to Render

### Option 1: Quick Deploy Script
```bash
./quick-deploy.sh
```
This script will guide you through the 2-step process automatically.

### Option 2: Manual 2-Step Process

**Step 1: Push to GitHub**
```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Deploy to Render"

# Add your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/spacial-ocr-client.git
git branch -M main
git push -u origin main
```

**Step 2: Deploy to Render**
1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select your repository
5. Render auto-detects settings from `render.yaml`
6. Click "Deploy Web Service"

### Option 3: One-Click Deploy
Click the "Deploy to Render" button above (requires GitHub repository setup first).

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the client application:
```bash
python client_app.py
```

3. Open your browser and go to `http://localhost:7860`

## API Endpoints

The client connects to the SPaCial OCR API at:
- **Base URL**: `https://cooldan-spacial-server-api.hf.space`
- **Process Endpoint**: `/ocr/process`
- **Documentation**: `/docs`

## Usage

1. **Upload Image**: Click on the image input area to upload an image
2. **Configure Options**:
   - **OCR Mode**: Choose between "fast" (position detection) or "accurate" (hard text search)
   - **Rotation**: Adjust image rotation in 90-degree increments
3. **Process**: Click "🚀 Process Image" to send the image to the API
4. **View Results**: 
   - See the processed image with bounding boxes
   - Read the extracted text with confidence scores and type indicators

## API Response Format

The API returns JSON with the following structure:
```json
{
  "zones": [
    {
      "text": "extracted text",
      "confidence": 0.95,
      "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 40},
      "is_dimension": true
    }
  ]
}
```

## Error Handling

The client includes comprehensive error handling for:
- Network connectivity issues
- API server errors
- Invalid image formats
- JSON parsing errors

## Based On

This client is based on the original implementation from:
- **UI Space**: https://huggingface.co/spaces/cooldan/SPaCial_ui
- **Server API**: https://huggingface.co/spaces/cooldan/SPaCial_server_api
