# SPaCial OCR Client

A client application to interact with the SPaCial AI OCR Service hosted on Hugging Face Spaces.

## Features

- 📝 Regular text detection (green bounding boxes)
- 📐 Dimension detection (blue bounding boxes) 
- ⚡ Fast and accurate OCR modes
- 🔄 Image rotation support
- 🖼️ Interactive web interface with Gradio

## Setup

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