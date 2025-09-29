#!/usr/bin/env python3
"""
Client application for SPaCial AI OCR Service
Based on the Hugging Face Space implementation
"""

import gradio as gr
import requests
from PIL import Image, ImageDraw
import io
import json
from typing import Tuple, Dict, Any

# API URL for the SPaCial OCR server
API_URL = "https://cooldan-spacial-server-api.hf.space"

def client_process_ocr(pil_img: Image.Image, mode: str = "fast", rotation: int = 0) -> Tuple[Image.Image, str]:
    """
    Function that sends the image to the FastAPI server for OCR processing.
    
    Args:
        pil_img: PIL Image object
        mode: OCR mode ('fast' for position detection, 'accurate' for hard text search)
        rotation: Rotation angle in degrees
    
    Returns:
        Tuple of (processed_image, extracted_text)
    """
    
    # 1. Prepare the Image for POST Request
    img_byte_arr = io.BytesIO()
    # Save the image in JPEG format in memory
    pil_img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Prepare data for submission (multipart/form-data)
    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
    
    # The processing endpoint is "/ocr/process"
    endpoint_url = f"{API_URL}/ocr/process"
    
    # Add query parameters
    params = {
        'mode': mode,
        'rotation': rotation
    }
    
    try:
        gr.Info("Sending image to OCR server...")
        
        # 2. Send HTTP POST Request
        response = requests.post(endpoint_url, files=files, params=params)
        
        # Check status code
        if response.status_code != 200:
            return pil_img, f"Server Error ({response.status_code}): {response.text}"

        # 3. Process JSON Response
        result_data = response.json()
        
        # Handle different response formats
        if isinstance(result_data, list):
            # If response is directly a list of zones
            zones = result_data
        else:
            # If response is a dict with zones key
            zones = result_data.get("zones", [])
        
        # 4. Draw Bounding Boxes on Image
        draw_img = pil_img.copy()
        draw = ImageDraw.Draw(draw_img)
        full_text_output = []
        
        for zone in zones:
            # Handle both dict and list zone formats
            if isinstance(zone, dict):
                text = zone.get("text", "")
                confidence = zone.get("confidence", 0.0)
                bbox = zone.get("bbox", {})
                is_dimension = zone.get("is_dimension", False)
            else:
                # If zone is a list, skip it
                continue
            
            # Text formatting for output
            prefix = "📐" if is_dimension else "📝"
            full_text_output.append(f"{prefix} [{confidence:.2f}] {text}")
            
            # Draw BBox
            if bbox:
                # Handle both dict and list bbox formats
                if isinstance(bbox, dict):
                    x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
                    x2, y2 = bbox.get("x2", 0), bbox.get("x2", 0)
                elif isinstance(bbox, list) and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                else:
                    continue
                
                # Choose color: green for regular text, blue for dimensions
                color = (0, 255, 0) if not is_dimension else (0, 0, 255)
                
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                
        gr.Info("Processing completed.")
        
        # Return the drawn image and text
        return draw_img, "\n".join(full_text_output)

    except requests.exceptions.RequestException as e:
        return pil_img, f"API Connection Error: Make sure the server ({API_URL}) is active.\nDetail: {str(e)}"

    except json.JSONDecodeError:
        return pil_img, f"Error: Invalid response (non-JSON) from server. Details: {response.text}"

    except Exception as e:
        return pil_img, f"Unexpected error: {str(e)}"

def check_api_status() -> str:
    """Check if the API server is running and accessible."""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            return f"✅ API Status: Running\nService: SPaCial AI OCR Service\nVersion: 1.0.0"
        else:
            return f"❌ API Error: {response.status_code}"
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

# --- Gradio Interface ---

with gr.Blocks(title="SPaCial OCR Client") as demo:
    gr.Markdown(
        """
        # 🖼️ SPaCial OCR Client
        This interface sends your image to the **FastAPI** server (`cooldan/SPaCial_server_api`)
        and displays the results with bounding boxes and extracted text.
        
        **Features:**
        - 📝 Regular text detection (green boxes)
        - 📐 Dimension detection (blue boxes)
        - ⚡ Fast and accurate OCR modes
        - 🔄 Image rotation support
        """
    )
    
    # API Status Check
    with gr.Row():
        status_btn = gr.Button("🔍 Check API Status", variant="secondary")
        status_output = gr.Textbox(label="API Status", interactive=False)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil", 
                label="1. Upload Image", 
                sources=["upload"], 
                interactive=True
            )
            
            # OCR Options
            with gr.Row():
                mode_dropdown = gr.Dropdown(
                    choices=["fast", "accurate"],
                    value="fast",
                    label="OCR Mode",
                    info="Fast: position detection, Accurate: hard text search"
                )
                rotation_slider = gr.Slider(
                    minimum=0,
                    maximum=360,
                    value=0,
                    step=90,
                    label="Rotation (degrees)"
                )
            
            process_btn = gr.Button("2. 🚀 Process Image", variant="primary")
        
        with gr.Column():
            # Output columns
            image_output = gr.Image(
                type="pil", 
                label="Result with Bounding Boxes", 
                interactive=False
            )
            
            text_output = gr.Textbox(
                label="3. Extracted Text (Confidence & Type)", 
                lines=20,
                interactive=False
            )
    
    # Event handlers
    status_btn.click(
        fn=check_api_status,
        outputs=status_output
    )
    
    process_btn.click(
        fn=client_process_ocr,
        inputs=[image_input, mode_dropdown, rotation_slider],
        outputs=[image_output, text_output]
    )

if __name__ == "__main__":
    # Launch the Gradio interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )