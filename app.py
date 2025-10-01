#!/usr/bin/env python3
"""
Hugging Face Spaces deployment for SPaCial AI OCR Service
FastAPI HTTP Server for OCR Service using PaddleOCR
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import cv2
import numpy as np
import os
import tempfile
import math
from pathlib import Path
import logging
import requests
import json
import base64
from datetime import datetime

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Telegram Bot Configuration (from environment variables)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Set environment variables BEFORE importing PaddleOCR
os.environ['PADDLE_PDX_MODEL_SOURCE'] = 'BOS'
os.environ['PADDLE_HOME'] = '/app/.paddlex'
os.environ['PADDLEX_HOME'] = '/app/.paddlex'
os.environ['PADDLEOCR_HOME'] = '/app/.paddleocr'
os.environ['TEMP'] = '/app/temp'
os.environ['TMP'] = '/app/temp'
os.environ['TMPDIR'] = '/app/temp'
os.environ['HOME'] = '/app'
os.environ['USER'] = 'app'

# Create directories with proper permissions for all PaddleOCR models
try:
    os.makedirs('/app/.paddlex', mode=0o777, exist_ok=True)
    os.makedirs('/app/.paddleocr', mode=0o777, exist_ok=True)
    os.makedirs('/app/.paddleocr/whl', mode=0o777, exist_ok=True)
    
    # Detection model
    os.makedirs('/app/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer', mode=0o777, exist_ok=True)
    
    # Recognition model
    os.makedirs('/app/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer', mode=0o777, exist_ok=True)
    
    # Classification model
    os.makedirs('/app/.paddleocr/whl/cls/en_ppocr_mobile_v2.0_cls_infer', mode=0o777, exist_ok=True)
    
    os.makedirs('/app/temp', mode=0o777, exist_ok=True)
    logger.info("Successfully created all PaddleOCR model directories with write permissions")
except Exception as e:
    logger.error(f"Failed to create directories: {e}")

# Now import PaddleOCR
from paddleocr import PaddleOCR


# Pydantic models for correction submission
class Zone(BaseModel):
    id: str
    text: str
    confidence: float
    bbox: List[float]
    x: float
    y: float
    width: float
    height: float
    orientation: int
    rotation: int
    corrected: Optional[bool] = False
    correction_type: Optional[str] = None  # 'text_fixed', 'box_moved', 'new_zone', 'deleted', 'validated'

class CorrectionSubmission(BaseModel):
    image_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[str] = None
    original_zones: List[Zone]
    corrected_zones: List[Zone]
    image_base64: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# Initialize FastAPI app
app = FastAPI(
    title="SPaCial AI OCR Service",
    description="OCR service for dimension detection using PaddleOCR",
    version="1.0.0"
)

# Add CORS middleware for HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for HF Spaces
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for mini app
try:
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        logger.info(f"Static files mounted from: {static_path}")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Global OCR instance (initialize once at startup)
ocr = None


# Telegram Bot Functions
def send_telegram_message(text: str) -> bool:
    """Send text message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram bot not configured (missing token or chat_id)")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Telegram message sent successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def send_telegram_photo(photo_data: bytes, caption: str = "") -> bool:
    """Send photo to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram bot not configured (missing token or chat_id)")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("image.jpg", photo_data, "image/jpeg")}
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML"
        }
        response = requests.post(url, files=files, data=data, timeout=30)
        response.raise_for_status()
        logger.info("Telegram photo sent successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram photo: {e}")
        return False


def send_telegram_document(document_data: bytes, filename: str, caption: str = "") -> bool:
    """Send document to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram bot not configured (missing token or chat_id)")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        files = {"document": (filename, document_data, "application/json")}
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML"
        }
        response = requests.post(url, files=files, data=data, timeout=30)
        response.raise_for_status()
        logger.info("Telegram document sent successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram document: {e}")
        return False


def format_correction_message(submission: CorrectionSubmission) -> str:
    """Format correction data into readable Telegram message"""
    timestamp = submission.timestamp or datetime.now().isoformat()
    image_id = submission.image_id or "unknown"
    user_id = submission.user_id or "anonymous"
    
    # Count changes
    total_original = len(submission.original_zones)
    total_corrected = len(submission.corrected_zones)
    changes = []
    
    # Analyze corrections
    for zone in submission.corrected_zones:
        if zone.correction_type:
            changes.append(zone.correction_type)
    
    text_fixed = changes.count('text_fixed')
    box_moved = changes.count('box_moved')
    new_zones = changes.count('new_zone')
    deleted = total_original - total_corrected if total_corrected < total_original else 0
    validated = changes.count('validated')
    
    message = f"""
📊 <b>New Training Data Received</b>

🆔 Image ID: <code>{image_id}</code>
👤 User: <code>{user_id}</code>
⏰ Time: <code>{timestamp}</code>

📈 <b>Statistics:</b>
• Original zones: {total_original}
• Corrected zones: {total_corrected}
• Text fixed: {text_fixed}
• Boxes moved: {box_moved}
• New zones added: {new_zones}
• Zones deleted: {deleted}
• Validated (OK): {validated}

🎯 <b>Changes Summary:</b>
"""
    
    # Show specific changes
    change_count = 0
    for i, corrected in enumerate(submission.corrected_zones):
        if corrected.correction_type and corrected.correction_type != 'validated':
            change_count += 1
            if change_count <= 10:  # Limit to 10 changes to avoid message too long
                original_text = ""
                # Find original zone
                for orig in submission.original_zones:
                    if orig.id == corrected.id:
                        original_text = orig.text
                        break
                
                if corrected.correction_type == 'text_fixed':
                    message += f"\n• <b>Text Fixed:</b> '{original_text}' → '{corrected.text}'"
                elif corrected.correction_type == 'box_moved':
                    message += f"\n• <b>Box Moved:</b> '{corrected.text}' @ ({int(corrected.x)}, {int(corrected.y)})"
                elif corrected.correction_type == 'new_zone':
                    message += f"\n• <b>New Zone:</b> '{corrected.text}' @ ({int(corrected.x)}, {int(corrected.y)})"
    
    if change_count > 10:
        message += f"\n... and {change_count - 10} more changes"
    
    message += "\n\n✅ <b>Data saved for model training</b>"
    
    return message


def send_correction_to_telegram(submission: CorrectionSubmission) -> bool:
    """Send correction data to Telegram channel/group"""
    try:
        # Format and send message
        message = format_correction_message(submission)
        message_sent = send_telegram_message(message)
        
        if not message_sent:
            return False
        
        # Send image if provided
        if submission.image_base64:
            try:
                # Decode base64 image
                image_data = base64.b64decode(submission.image_base64)
                caption = f"Blueprint: {submission.image_id or 'unknown'}"
                send_telegram_photo(image_data, caption)
            except Exception as e:
                logger.error(f"Failed to send image: {e}")
        
        # Send JSON data as document for training
        json_data = {
            "image_id": submission.image_id,
            "user_id": submission.user_id,
            "timestamp": submission.timestamp,
            "original_zones": [zone.dict() for zone in submission.original_zones],
            "corrected_zones": [zone.dict() for zone in submission.corrected_zones],
            "metadata": submission.metadata
        }
        json_bytes = json.dumps(json_data, indent=2).encode('utf-8')
        filename = f"training_data_{submission.image_id or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        send_telegram_document(json_bytes, filename, "Training data (JSON)")
        
        logger.info(f"Correction data sent to Telegram successfully for image: {submission.image_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending correction to Telegram: {e}")
        return False


def send_telegram_reply(chat_id: int, text: str, reply_to_message_id: Optional[int] = None) -> bool:
    """Send reply message to specific chat"""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram bot not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send reply: {e}")
        return False


def send_message_with_webapp_button(chat_id: int, text: str, reply_to_message_id: Optional[int] = None) -> bool:
    """Send message with web app button to open mini app"""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram bot not configured")
        return False
    
    try:
        # Get the current host URL from environment or use default
        webapp_url = os.getenv('SPACE_HOST', 'https://your-space.hf.space')
        if not webapp_url.startswith('http'):
            webapp_url = f'https://{webapp_url}'
        webapp_url = f'{webapp_url}/camera'
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "reply_markup": {
                "inline_keyboard": [
                    [
                        {
                            "text": "📸 Live Camera Scanner",
                            "web_app": {"url": webapp_url}
                        }
                    ],
                    [
                        {
                            "text": "📁 Upload Photo Instead",
                            "callback_data": "upload_photo"
                        }
                    ]
                ]
            }
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send message with webapp button: {e}")
        return False


def send_telegram_photo_to_chat(chat_id: int, photo_data: bytes, caption: str = "", reply_markup: Optional[Dict] = None) -> bool:
    """Send photo to specific chat with optional keyboard"""
    if not TELEGRAM_BOT_TOKEN:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("image.jpg", photo_data, "image/jpeg")}
        data = {
            "chat_id": chat_id,
            "caption": caption,
            "parse_mode": "HTML"
        }
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        
        response = requests.post(url, files=files, data=data, timeout=30)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send photo: {e}")
        return False


def download_telegram_file(file_id: str) -> Optional[bytes]:
    """Download file from Telegram"""
    if not TELEGRAM_BOT_TOKEN:
        return None
    
    try:
        # Get file path
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile"
        response = requests.get(url, params={"file_id": file_id}, timeout=10)
        response.raise_for_status()
        file_path = response.json()["result"]["file_path"]
        
        # Download file
        download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        file_response = requests.get(download_url, timeout=30)
        file_response.raise_for_status()
        
        return file_response.content
    except Exception as e:
        logger.error(f"Failed to download file: {e}")
        return None


def format_ocr_results_for_telegram(zones: list) -> str:
    """Format OCR results as readable text for Telegram"""
    if not zones:
        return "❌ No text detected in the image."
    
    message = f"📊 <b>OCR Results</b>\n\n"
    message += f"✅ Found {len(zones)} text zones:\n\n"
    
    for i, zone in enumerate(zones, 1):
        text = zone.get('text', '')
        confidence = zone.get('confidence', 0)
        x = int(zone.get('x', 0))
        y = int(zone.get('y', 0))
        
        message += f"{i}. <code>{text}</code>\n"
        message += f"   📍 Position: ({x}, {y}) | Confidence: {confidence:.0%}\n\n"
        
        if i >= 20:  # Limit to 20 zones to avoid message too long
            remaining = len(zones) - 20
            message += f"... and {remaining} more zones\n"
            break
    
    return message


def create_inline_keyboard() -> Dict:
    """Create inline keyboard for OCR results"""
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "✅ All Correct", "callback_data": "validate_all"},
                {"text": "✏️ Need Corrections", "callback_data": "need_corrections"}
            ],
            [
                {"text": "📥 Download JSON", "callback_data": "download_json"},
                {"text": "🔄 Process Again", "callback_data": "reprocess"}
            ]
        ]
    }
    return keyboard


async def process_telegram_photo(photo_file_id: str, chat_id: int, message_id: int):
    """Process photo sent to Telegram bot"""
    try:
        # Send processing message
        send_telegram_reply(chat_id, "⏳ Processing your blueprint... Please wait.", message_id)
        
        # Download photo
        photo_data = download_telegram_file(photo_file_id)
        if not photo_data:
            send_telegram_reply(chat_id, "❌ Failed to download image. Please try again.", message_id)
            return
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(photo_data)
            temp_path = temp_file.name
        
        # Process with OCR
        result = process_image(temp_path, mode="fast", rotation=0)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        zones = result.get('zones', [])
        
        # Format and send results
        results_text = format_ocr_results_for_telegram(zones)
        send_telegram_reply(chat_id, results_text, message_id)
        
        # Send overlay image if available
        overlay_path = result.get('metadata', {}).get('overlay_path')
        if overlay_path and os.path.exists(overlay_path):
            with open(overlay_path, 'rb') as f:
                overlay_data = f.read()
            
            keyboard = create_inline_keyboard()
            send_telegram_photo_to_chat(
                chat_id, 
                overlay_data, 
                "Blueprint with detected text zones",
                keyboard
            )
            
            # Clean up overlay
            os.unlink(overlay_path)
        
        # Send JSON data
        json_data = json.dumps(result, indent=2).encode('utf-8')
        filename = f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        files = {"document": (filename, json_data, "application/json")}
        data = {"chat_id": chat_id, "caption": "📄 Complete OCR results (JSON)"}
        requests.post(url, files=files, data=data, timeout=30)
        
        logger.info(f"Successfully processed Telegram photo for chat: {chat_id}")
        
    except Exception as e:
        logger.error(f"Error processing Telegram photo: {e}")
        send_telegram_reply(chat_id, f"❌ Error processing image: {str(e)}", message_id)


def handle_telegram_command(command: str, chat_id: int, message_id: int):
    """Handle Telegram bot commands"""
    if command == "/start":
        message = """
🤖 <b>Welcome to SPaCial AI OCR Bot!</b>

I can help you extract dimensions and text from blueprints and technical drawings.

<b>How to use:</b>
1. 📸 Use Live Camera Scanner (button below)
2. Or send me a blueprint image
3. I'll detect all text zones
4. Review and correct if needed
5. Download results as JSON

<b>Commands:</b>
/help - Show this help message
/status - Check OCR service status

Choose an option below to get started! 👇
"""
        # Send message with mini app button
        send_message_with_webapp_button(chat_id, message, message_id)
    
    elif command == "/help":
        message = """
📖 <b>Help - SPaCial AI OCR Bot</b>

<b>Basic Usage:</b>
• Send a blueprint image → Get OCR results
• Use buttons to validate or correct
• Download JSON for further processing

<b>Tips for Best Results:</b>
• Use clear, high-quality images
• Ensure good lighting and contrast
• Avoid excessive blur or distortion

<b>Correction Flow:</b>
1. Review detected text
2. Click "Need Corrections" if needed
3. Reply with corrections
4. Data saved for model training

<b>Commands:</b>
/start - Welcome message
/help - This help message
/status - Service status

Questions? Send feedback to the development team!
"""
        send_telegram_reply(chat_id, message, message_id)
    
    elif command == "/status":
        service_status = "✅ Online" if ocr is not None else "❌ Offline"
        message = f"""
📊 <b>Service Status</b>

🤖 OCR Service: {service_status}
🔧 Version: 1.0.0
⚡ Mode: {"GPU" if ocr else "CPU"}

Ready to process blueprints!
"""
        send_telegram_reply(chat_id, message, message_id)
    
    else:
        send_telegram_reply(chat_id, "❓ Unknown command. Use /help for available commands.", message_id)


def is_dimension_text(text):
    """Check if text looks like a dimension (number with optional tolerance)"""
    import re
    # Pattern for dimensions: number with optional tolerance (±0.1, +0.2, -0.3, etc.)
    dimension_pattern = r'^\d+(?:\.\d+)?(?:\s*[±+\-]\s*\d+(?:\.\d+)?)?$'
    return bool(re.match(dimension_pattern, text.strip()))

def parse_tolerance(text):
    """Parse tolerance information from dimension text"""
    import re
    # Look for tolerance patterns: ±0.1, +0.2, -0.3
    tolerance_match = re.search(r'([±+\-])\s*(\d+(?:\.\d+)?)', text)
    if tolerance_match:
        sign = tolerance_match.group(1)
        value = float(tolerance_match.group(2))
        if sign == '±':
            return {"plus": value, "minus": value}
        elif sign == '+':
            return {"plus": value, "minus": 0}
        elif sign == '-':
            return {"plus": 0, "minus": value}
    return {"plus": 0, "minus": 0}

def merge_vertical_text(zones):
    """Merge nearby vertical text that might have been split by OCR"""
    if len(zones) < 2:
        return zones
    
    merged_zones = []
    used_indices = set()
    
    for i, zone1 in enumerate(zones):
        if i in used_indices:
            continue
            
        bbox1 = zone1.get('bbox', [])
        if len(bbox1) < 4:
            merged_zones.append(zone1)
            continue
            
        x1_1, y1_1, x2_1, y2_1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        text1 = zone1.get('text', '')
        
        # Look for nearby zones to merge
        merged_text = text1
        merged_bbox = [x1_1, y1_1, x2_1, y2_1]
        
        for j, zone2 in enumerate(zones[i+1:], i+1):
            if j in used_indices:
                continue
                
            bbox2 = zone2.get('bbox', [])
            if len(bbox2) < 4:
                continue
                
            x1_2, y1_2, x2_2, y2_2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
            text2 = zone2.get('text', '')
            
            # Check if zones are vertically close and horizontally aligned
            vertical_distance = abs((y1_1 + y2_1) / 2 - (y1_2 + y2_2) / 2)
            horizontal_overlap = not (x2_1 < x1_2 or x2_2 < x1_1)
            
            # If zones are close vertically and overlap horizontally, merge them
            if vertical_distance < 30 and horizontal_overlap:
                # Merge text
                if y1_2 < y1_1:  # zone2 is above zone1
                    merged_text = text2 + text1
                    merged_bbox = [min(x1_1, x1_2), min(y1_1, y1_2), 
                                 max(x2_1, x2_2), max(y2_1, y2_2)]
                else:  # zone1 is above zone2
                    merged_text = text1 + text2
                    merged_bbox = [min(x1_1, x1_2), min(y1_1, y1_2), 
                                 max(x2_1, x2_2), max(y2_1, y2_2)]
                
                used_indices.add(j)
                logger.info(f"Merged vertical text: '{text1}' + '{text2}' = '{merged_text}'")
        
        # Create merged zone
        merged_zone = {
            "id": f"ocr_zone_{len(merged_zones)}",
            "text": merged_text,
            "confidence": zone1.get('confidence', 0.9),
            "bbox": merged_bbox,
            "x": float(merged_bbox[0]),
            "y": float(merged_bbox[1]),
            "width": float(merged_bbox[2] - merged_bbox[0]),
            "height": float(merged_bbox[3] - merged_bbox[1]),
            "orientation": zone1.get('orientation', 0),
            "rotation": 0
        }
        
        merged_zones.append(merged_zone)
        used_indices.add(i)
    
    # Add any remaining zones that weren't merged
    for i, zone in enumerate(zones):
        if i not in used_indices:
            merged_zones.append(zone)
    
    logger.info(f"Merged {len(zones)} zones into {len(merged_zones)} zones")
    return merged_zones

def create_overlay_image(image_path, zones):
    """Create overlay image with bounding boxes (using direct coordinates, no rotation)"""
    try:
        # Load the original image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Create a copy for overlay
        overlay = img.copy()
        
        for zone in zones:
            bbox = zone.get('bbox', [])
            if bbox and len(bbox) >= 4:
                # Handle both list format [x1, y1, x2, y2] and dict format
                if isinstance(bbox, list):
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                else:
                    x1 = bbox.get('x1', 0)
                    y1 = bbox.get('y1', 0)
                    x2 = bbox.get('x2', 0)
                    y2 = bbox.get('y2', 0)
                
                # Draw bounding box
                #cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if all(isinstance(coord, (int, float)) and not math.isnan(coord) for coord in [x1, y1, x2, y2]):
                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                
                # Add text label
                text = zone.get('text', '')
                if text:
                    #cv2.putText(overlay, text, (x1, y1 - 10), 
                    #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
                    if all(isinstance(coord, (int, float)) and not math.isnan(coord) for coord in [x1, y1]):
                        text_x, text_y = int(x1), int(y1 - 10)
                        if text_y > 0:  # Ensure text is not above image
                            cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


                    
        # Save overlay image
        overlay_path = str(image_path).replace('.jpg', '_overlay.jpg')
        cv2.imwrite(overlay_path, overlay)
        return overlay_path
        
    except Exception as e:
        logger.error(f"Error creating overlay: {e}")
        return None

def process_image(image_path, mode="fast", rotation=0):
    """Process a single image and return OCR results with specified mode"""
    try:
        # Get image dimensions for coordinate transformation
        img = cv2.imread(str(image_path))
        height, width = img.shape[:2]
        
        # Apply rotation if specified
        if rotation != 0:
            # Rotate the image
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (width, height))
            
            # Save rotated image temporarily
            rotated_path = str(image_path).replace('.jpg', '_rotated.jpg')
            cv2.imwrite(rotated_path, img)
            image_path = rotated_path
        
        # Use single OCR model (mode parameter kept for future use)
        logger.info(f"Processing image with OCR mode: {mode}")
        
        # Use the same logic as working local server
        logger.info(f"Running OCR on image: {image_path}")
        result = ocr.ocr(str(image_path))
        logger.info(f"OCR result type: {type(result)}, length: {len(result) if result else 'None'}")
        
        zones = []
        if result and len(result) > 0:
            logger.info(f"Processing {len(result)} result lines")
            # PaddleOCR returns: [[[bbox, text, confidence]], ...]
            for line_idx, line in enumerate(result):
                logger.info(f"Line {line_idx}: {type(line)}, length: {len(line) if line else 'None'}")
                if line:
                    for item_idx, item in enumerate(line):
                        logger.info(f"  Item {item_idx}: {type(item)}, length: {len(item) if item else 'None'}")
                        # Handle different result formats
                        if len(item) == 3:
                            bbox, text, confidence = item
                            logger.info(f"    Found 3-item format: bbox={bbox}, text='{text}' (type: {type(text)}), confidence={confidence}")
                        elif len(item) == 2:
                            bbox, text = item
                            confidence = 0.9  # Default confidence
                            logger.info(f"    Found 2-item format: bbox={bbox}, text='{text}' (type: {type(text)}), confidence={confidence}")
                        else:
                            logger.warning(f"    Unexpected item format with {len(item)} elements: {item}")
                            continue
                        
                        # Handle text that might be a tuple or list
                        if isinstance(text, (tuple, list)):
                            if len(text) >= 2:
                                text = text[0]  # Take the first element (usually the text)
                                logger.info(f"    Extracted text from tuple/list: '{text}'")
                            else:
                                logger.warning(f"    Empty tuple/list for text: {text}")
                                continue
                        
                        if isinstance(text, str) and text.strip():
                            logger.info(f"    Processing text: '{text}'")
                            # Extract bounding box coordinates
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            
                            # Simple orientation detection based on bbox shape
                            width = x2 - x1
                            height = y2 - y1
                            text_orientation = 0
                            
                            # Basic orientation detection
                            if height > width * 1.5:  # Vertical text
                                text_orientation = 90
                            elif abs(height - width) < max(height, width) * 0.3:  # Roughly square
                                # Could be 45°, but keep as 0° for now
                                text_orientation = 0
                            
                            zone = {
                                "id": f"ocr_zone_{len(zones)}",
                                "text": text,
                                "confidence": float(confidence),
                                "bbox": [x1, y1, x2, y2],
                                "x": float(x1),
                                "y": float(y1),
                                "width": float(x2 - x1),
                                "height": float(y2 - y1),
                                "orientation": text_orientation,
                                "rotation": 0  # Box rotation (keep at 0 for now)
                            }
                            zones.append(zone)
        
        # Post-process to merge nearby vertical text that might have been split
        zones = merge_vertical_text(zones)
        
        # Create overlay image with bounding boxes
        overlay_path = create_overlay_image(image_path, zones)
        
        # Clean up rotated image if it was created
        if rotation != 0 and os.path.exists(image_path):
            os.unlink(image_path)
        
        return {
            "zones": zones,
            "metadata": {
                "total_zones": len(zones),
                "overlay_path": overlay_path,
                "detected_angle": 0  # No rotation - detect as-is
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {
            "zones": [],
            "metadata": {
                "total_zones": 0,
                "overlay_path": None,
                "detected_angle": 0,
                "error": str(e)
            }
        }

@app.on_event("startup")
async def startup_event():
    """Initialize PaddleOCR models on startup"""
    global ocr
    logger.info("Initializing PaddleOCR model...")
    
    # Single OCR instance (try GPU first, fallback to CPU)
    try:
        logger.info("Attempting to initialize PaddleOCR with GPU...")
        ocr = PaddleOCR(
            use_doc_orientation_classify=True,  # Enable document orientation
            use_doc_unwarping=False,
            use_textline_orientation=True,  # Enable text line orientation
            use_gpu=True,  # Try GPU first
            lang='en'
        )
        logger.info("PaddleOCR initialized successfully with GPU!")
    except Exception as gpu_error:
        logger.warning(f"GPU initialization failed: {gpu_error}")
        logger.info("Falling back to CPU mode...")
        ocr = PaddleOCR(
            use_doc_orientation_classify=True,  # Enable document orientation
            use_doc_unwarping=False,
            use_textline_orientation=True,  # Enable text line orientation
            use_gpu=False,  # Fallback to CPU
            lang='en'
        )
        logger.info("PaddleOCR initialized successfully with CPU!")
    
    logger.info("PaddleOCR initialized successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "SPaCial AI OCR Service",
        "status": "running",
        "ocr_initialized": ocr is not None,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "service": "SPaCial AI OCR Service",
        "version": "1.0.0",
        "endpoints": {
            "ocr": "/ocr/process",
            "corrections": "/corrections/submit",
            "telegram_status": "/telegram/status",
            "telegram_webhook": "/telegram/webhook",
            "camera_app": "/camera"
        }
    }

@app.get("/camera", response_class=HTMLResponse)
async def camera_app():
    """Serve the mini app camera interface"""
    try:
        camera_html_path = Path(__file__).parent / "static" / "camera.html"
        if camera_html_path.exists():
            with open(camera_html_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(
                content="<h1>Camera app not found</h1><p>Please ensure static/camera.html exists.</p>",
                status_code=404
            )
    except Exception as e:
        logger.error(f"Error serving camera app: {e}")
        return HTMLResponse(
            content=f"<h1>Error</h1><p>{str(e)}</p>",
            status_code=500
        )

@app.post("/ocr/process")
async def process_ocr(file: UploadFile = File(...), mode: str = Query("fast", description="OCR mode: 'fast' for position detection, 'accurate' for hard text search"), rotation: int = Query(0, description="Rotation angle in degrees")):
    """
    Process uploaded image with OCR
    Returns detected zones with bounding boxes and text
    """
    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Process image with specified mode and rotation
        result = process_image(temp_path, mode, rotation)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up temporary file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/ocr/process-path")
async def process_ocr_path(image_path: str = Query(...), mode: str = Query("fast", description="OCR mode: 'fast' for position detection, 'accurate' for hard text search")):
    """
    Process image from file path
    Returns detected zones with bounding boxes and text
    """
    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        result = process_image(image_path, mode)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing image path: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/corrections/submit")
async def submit_corrections(submission: CorrectionSubmission):
    """
    Submit user corrections for training data collection
    Sends data to Telegram bot for storage and future model training
    
    Expected format:
    {
        "image_id": "unique_image_id",
        "user_id": "user_identifier",
        "timestamp": "2025-09-30T12:00:00",
        "original_zones": [...],  // OCR predictions
        "corrected_zones": [...], // User corrections
        "image_base64": "base64_encoded_image",
        "metadata": {...}
    }
    
    Each zone should have:
    - correction_type: 'text_fixed', 'box_moved', 'new_zone', 'deleted', 'validated'
    """
    try:
        # Add timestamp if not provided
        if not submission.timestamp:
            submission.timestamp = datetime.now().isoformat()
        
        # Send to Telegram
        success = send_correction_to_telegram(submission)
        
        if success:
            return JSONResponse(content={
                "status": "success",
                "message": "Correction data submitted successfully",
                "telegram_sent": True,
                "image_id": submission.image_id,
                "timestamp": submission.timestamp
            })
        else:
            # Still return success even if Telegram fails (bot might not be configured)
            logger.warning("Telegram submission failed, but request processed")
            return JSONResponse(content={
                "status": "success",
                "message": "Correction data received (Telegram delivery failed - check bot configuration)",
                "telegram_sent": False,
                "image_id": submission.image_id,
                "timestamp": submission.timestamp
            })
        
    except Exception as e:
        logger.error(f"Error submitting corrections: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting corrections: {str(e)}")

@app.get("/telegram/status")
async def telegram_status():
    """Check if Telegram bot is configured"""
    configured = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    
    status = {
        "configured": configured,
        "bot_token_set": bool(TELEGRAM_BOT_TOKEN),
        "chat_id_set": bool(TELEGRAM_CHAT_ID)
    }
    
    if configured:
        # Test connection
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bot_info = response.json()
                status["connection"] = "ok"
                status["bot_username"] = bot_info.get("result", {}).get("username")
            else:
                status["connection"] = "failed"
                status["error"] = "Invalid bot token"
        except Exception as e:
            status["connection"] = "failed"
            status["error"] = str(e)
    else:
        status["connection"] = "not_configured"
        status["message"] = "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables"
    
    return JSONResponse(content=status)

@app.post("/telegram/webhook")
async def telegram_webhook(request: dict = Body(...)):
    """
    Telegram bot webhook endpoint
    Handles incoming messages, photos, commands, and callbacks
    """
    try:
        logger.info(f"Received webhook: {json.dumps(request)}")
        
        # Handle regular messages
        if "message" in request:
            message = request["message"]
            chat_id = message.get("chat", {}).get("id")
            message_id = message.get("message_id")
            
            # Handle commands
            if "text" in message and message["text"].startswith("/"):
                command = message["text"].split()[0]
                handle_telegram_command(command, chat_id, message_id)
                return JSONResponse(content={"ok": True})
            
            # Handle photos
            if "photo" in message:
                # Get largest photo
                photos = message["photo"]
                largest_photo = max(photos, key=lambda p: p.get("file_size", 0))
                file_id = largest_photo["file_id"]
                
                # Process photo asynchronously
                await process_telegram_photo(file_id, chat_id, message_id)
                return JSONResponse(content={"ok": True})
            
            # Handle video
            if "video" in message:
                send_telegram_reply(
                    chat_id, 
                    "📹 Video processing coming soon! For now, please send individual photos or use the live camera scanner.",
                    message_id
                )
                return JSONResponse(content={"ok": True})
            
            # Handle documents (images as files)
            if "document" in message:
                doc = message["document"]
                mime_type = doc.get("mime_type", "")
                if mime_type.startswith("image/"):
                    file_id = doc["file_id"]
                    await process_telegram_photo(file_id, chat_id, message_id)
                    return JSONResponse(content={"ok": True})
            
            # Handle text messages
            if "text" in message:
                text = message["text"]
                # Could be a correction or feedback
                send_telegram_reply(
                    chat_id,
                    "👋 Send me a blueprint image to scan!\n\nOr use /help for more information.",
                    message_id
                )
                return JSONResponse(content={"ok": True})
        
        # Handle callback queries (button clicks)
        if "callback_query" in request:
            callback = request["callback_query"]
            callback_id = callback["id"]
            chat_id = callback["message"]["chat"]["id"]
            message_id = callback["message"]["message_id"]
            data = callback["data"]
            
            # Answer callback to remove loading state
            answer_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
            requests.post(answer_url, json={"callback_query_id": callback_id}, timeout=5)
            
            # Handle different callback actions
            if data == "validate_all":
                send_telegram_reply(
                    chat_id,
                    "✅ Great! All detections validated.\n\n📊 This data helps improve the model. Thank you!",
                    message_id
                )
            elif data == "need_corrections":
                send_telegram_reply(
                    chat_id,
                    "✏️ <b>How to correct:</b>\n\nReply with corrections in this format:\n<code>zone_number | correct_text</code>\n\nExample:\n<code>1 | 125.0</code>\n<code>3 | ±0.1</code>\n\nOr describe what's wrong!",
                    message_id
                )
            elif data == "download_json":
                send_telegram_reply(
                    chat_id,
                    "📥 JSON file was already sent above! Check the document.",
                    message_id
                )
            elif data == "reprocess":
                send_telegram_reply(
                    chat_id,
                    "🔄 To reprocess, please send the image again.",
                    message_id
                )
            elif data == "upload_photo":
                send_telegram_reply(
                    chat_id,
                    "📁 <b>Upload Photo Mode</b>\n\nSimply send me a photo of your blueprint and I'll process it!\n\nOr use /start to access the live camera scanner again.",
                    message_id
                )
            
            return JSONResponse(content={"ok": True})
        
        return JSONResponse(content={"ok": True})
        
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        return JSONResponse(content={"ok": False, "error": str(e)})

@app.post("/telegram/set-webhook")
async def set_telegram_webhook(webhook_url: str = Query(...)):
    """
    Set Telegram webhook URL
    Call this once after deployment to configure your bot
    
    Example: POST /telegram/set-webhook?webhook_url=https://your-space.hf.space/telegram/webhook
    """
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=400, detail="TELEGRAM_BOT_TOKEN not configured")
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
        response = requests.post(url, json={"url": webhook_url}, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        return JSONResponse(content={
            "success": True,
            "message": "Webhook set successfully",
            "webhook_url": webhook_url,
            "telegram_response": result
        })
    except Exception as e:
        logger.error(f"Failed to set webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set webhook: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
