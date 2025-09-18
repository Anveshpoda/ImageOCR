from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from InitFiles.signboard_extractor import SignboardExtractor

app = FastAPI(title="Signboard OCR API")

# Initialize the extractor
extractor = SignboardExtractor()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageURL(BaseModel):
    image_url: str

@app.post("/extract-date")
async def extract_date(image_data: ImageURL):
    """
    Extract date from an image URL.
    Expected input: {"image_url": "https://example.com/image.jpg"}
    """
    try:
        # Download image
        response = requests.get(image_data.image_url)
        response.raise_for_status()
        
        # Convert to OpenCV format
        image = Image.open(BytesIO(response.content))
        image_np = np.array(image)
        
        # Convert RGB to BGR if needed
        if image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Save to temp file (PaddleOCR works better with file paths)
        temp_path = "/tmp/temp_image.jpg"
        cv2.imwrite(temp_path, image_np)
        
        # Extract data
        result, all_text = extractor.extract_data(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "success": True,
            "extracted_date": result.get("date"),
            "all_detected_text": all_text
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
