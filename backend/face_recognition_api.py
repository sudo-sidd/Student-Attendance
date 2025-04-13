import torch
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from typing import List, Optional
from PIL import Image
import torchvision.transforms as transforms
from .LightCNN.light_cnn import LightCNN_29Layers_v2
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import uvicorn
import shutil
from pathlib import Path
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # allow your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = pd.read_csv('../NAME_LIST.csv')

SELECTED_CLASS = "A"
print(data.columns)
# Global variables to hold models (loaded once at startup)
face_model = None
yolo_model = None
gallery = None
device = None
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class RecognitionResult(BaseModel):
    face_id: int
    identity: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]

class RecognitionResponse(BaseModel):
    image_base64: str
    faces: List[RecognitionResult]
    status: str
    message: str

class AttendanceRecord(BaseModel):
    RegisterNumber: str
    FullName: str
    Department: str
    Status: str

class AttendanceData(BaseModel):
    section: str
    date: str
    records: List[AttendanceRecord]

@app.get('/sections')
async def get_sections():
    res = data.groupby("Section")
    return {
        "sections": list(res.groups.keys())
    }

# Update the get_class endpoint to reload the gallery when section changes
@app.get('/class/{section}')
async def get_class(section: str):
    global SELECTED_CLASS, gallery
    filtered_data = data[data["Section"] == section]
    print(SELECTED_CLASS)
    # Update selected class and reload gallery for the new section
    if SELECTED_CLASS != section:
        SELECTED_CLASS = section
        gallery = load_gallery_for_section(SELECTED_CLASS)
        print(f"Updated selected class to {SELECTED_CLASS} and reloaded gallery")
    
    return JSONResponse(content=filtered_data.to_dict(orient="records"))

# Add the helper function to load gallery dynamically based on section
def load_gallery_for_section(section: str):
    gallery_path = f"ML_{section}.pth"
    try:
        loaded_gallery = torch.load(gallery_path)
        print(f"✓ Gallery loaded for section {section} with {len(loaded_gallery)} identities")
        return loaded_gallery
    except Exception as e:
        print(f"❌ Error loading gallery for section {section}: {e}")
        return {}

@app.post("/save-attendance-csv")
async def save_attendance_csv(attendance_data: AttendanceData):
    # Create directory for storing attendance records if it doesn't exist
    output_dir = Path("attendance_records")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Format date for filename
    date_formatted = attendance_data.date.replace("-", "")
    
    # Create filename with section and date
    filename = f"attendance_{attendance_data.section}_{date_formatted}.csv"
    file_path = output_dir / filename
    
    # Convert records to DataFrame
    records_dict = [record.dict() for record in attendance_data.records]
    df = pd.DataFrame(records_dict)
    
    # Add timestamp
    df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    
    # Return file for download
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/csv"
    )

# Update the startup event to use the helper function
@app.on_event("startup") 
async def startup_event():
    """Load models and gallery at startup"""
    global face_model, yolo_model, gallery, device
    
    # Path configurations (customize these)
    model_path = "./checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar"
    yolo_path = "./yolo/weights/yolo11n-face.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for inference")
    
    # Load face recognition model
    face_model = LightCNN_29Layers_v2(num_classes=100)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Filter out the fc2 layer parameters
        if 'state_dict' in checkpoint:
            new_state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if 'fc2' in k:
                    continue
                new_k = k.replace("module.", "")
                new_state_dict[new_k] = v
        else:
            new_state_dict = {}
            for k, v in checkpoint.items():
                if 'fc2' in k:
                    continue
                new_k = k.replace("module.", "")
                new_state_dict[new_k] = v
        
        face_model.load_state_dict(new_state_dict, strict=False)
        face_model = face_model.to(device)
        face_model.eval()
        print("✓ Face recognition model loaded")
    except Exception as e:
        print(f"❌ Error loading face model: {e}")
        raise RuntimeError(f"Failed to load face recognition model: {e}")
    
    # Load YOLO model
    try:
        yolo_model = YOLO(yolo_path)
        print("✓ YOLO face detection model loaded")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    # Load gallery for the default section
    gallery = load_gallery_for_section(SELECTED_CLASS)

def process_image(image_bytes, threshold=0.45):
    """
    Process an image for face recognition with dynamic thresholding based on face size
    
    Args:
        image_bytes: Raw image bytes
        threshold: Base recognition confidence threshold
        
    Returns:
        Tuple of (processed_image, recognition_results)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Create a copy for drawing results
    result_img = img.copy()
    face_id = 0
    
    # Detect faces using YOLO
    results = yolo_model(img)
    
    # Get image dimensions to calculate relative face sizes
    img_height, img_width = img.shape[:2]
    img_area = img_height * img_width
    
    # First, collect all face data with complete matching information
    face_data = []
    
    for result in results:
        for box in result.boxes:
            face_id += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate original face size before padding
            orig_face_width = x2 - x1
            orig_face_height = y2 - y1
            face_area = orig_face_width * orig_face_height
            
            # Calculate face size ratio relative to the image
            face_ratio = face_area / img_area
            
            # Adjust threshold based on face size
            # Smaller faces get more lenient thresholds, larger faces get stricter thresholds
            if face_ratio < 0.01:  # Very small face (less than 1% of image)
                dynamic_threshold = max(threshold - 0.1, 0.2)  # Lower threshold but not below 0.2
            elif face_ratio < 0.03:  # Small face (1-3% of image)
                dynamic_threshold = max(threshold - 0.05, 0.3)  # Slightly lower threshold
            elif face_ratio > 0.15:  # Large face (>15% of image)
                dynamic_threshold = min(threshold + 0.1, 0.9)  # Higher threshold but not above 0.9
            else:
                dynamic_threshold = threshold  # Default threshold
            
            # Add padding around face - identical padding as in test_face_api
            h, w = img.shape[:2]
            face_w = x2 - x1
            face_h = y2 - y1
            pad_x = int(face_w * 0.2)
            pad_y = int(face_h * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            face = img[y1:y2, x1:x2]
            
            # Skip invalid faces
            if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                continue
            
            # Apply image enhancement techniques for better recognition
            # 1. Convert to grayscale
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_face = clahe.apply(gray_face)
            
            # 3. Apply slight Gaussian blur to reduce noise
            enhanced_face = cv2.GaussianBlur(enhanced_face, (3, 3), 0)
            
            # 4. Normalize the image
            enhanced_face = cv2.normalize(enhanced_face, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert to PIL image
            face_pil = Image.fromarray(enhanced_face)
            
            # Get face tensor and extract embedding
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, embedding = face_model(face_tensor)
                face_embedding = embedding.cpu().squeeze().numpy()
            
            # Find all potential matches above the dynamic threshold
            matches = []
            for identity, gallery_embedding in gallery.items():
                similarity = 1 - cosine(face_embedding, gallery_embedding)
                if similarity > dynamic_threshold:
                    matches.append((identity, similarity))
            
            # Sort matches by confidence (highest first)
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # Store all data for this face including the dynamic threshold
            face_data.append({
                'id': face_id,
                'bbox': (x1, y1, x2, y2),
                'matches': matches,
                'face_size': face_ratio,
                'threshold': dynamic_threshold
            })
    
    # Sort faces by their highest confidence score (descending)
    face_data.sort(key=lambda x: x['matches'][0][1] if x['matches'] else 0, reverse=True)
    
    # Assign identities without duplicates
    assigned_identities = set()
    face_results = []
    
    for data in face_data:
        face_id = data['id']
        x1, y1, x2, y2 = data['bbox']
        matches = data['matches']
        face_ratio = data['face_size']
        dynamic_threshold = data['threshold']
        
        # Try to find a unique match
        best_match = "Unknown"
        best_score = -1
        
        for identity, score in matches:
            if identity not in assigned_identities:
                best_match = identity
                best_score = score
                break
        
        # If we found a valid match, mark it as assigned
        if best_match != "Unknown":
            assigned_identities.add(best_match)
        
        # Draw result on image
        if best_match != "Unknown":
            # Known identity - green box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Include size & threshold information for debugging
            size_info = f"{face_ratio*100:.1f}%"
            label = f"{best_match} ({best_score:.2f}, {size_info})"
            cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Unknown - red box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Include size & threshold information for debugging
            size_info = f"{face_ratio*100:.1f}%"
            label = f"Unknown ({size_info}, thr:{dynamic_threshold:.2f})"
            cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add to results
        face_results.append(RecognitionResult(
            face_id=face_id,
            identity=best_match,
            confidence=best_score,
            bbox=[int(x1), int(y1), int(x2), int(y2)]
        ))
    
    # Save the result image
    cv2.imwrite('result.jpg', result_img)
    
    # Encode image to base64 to return in response
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64, face_results

@app.post("/upload-image/")
async def upload_and_recognize(
    file: UploadFile = File(...),
    threshold: float = Form(0.45)
):
    """
    Upload and process an image for face recognition.
    """
    # Check if models are loaded
    if face_model is None or yolo_model is None or gallery is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Please check server logs.")
    
    # Validate threshold
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
    
    try:
        # Read file content
        contents = await file.read()
        
        # Process the image and get results
        img_base64, faces = process_image(contents, threshold)
        
        # Extract just the IDs from recognized faces (last 3 digits)
        detected_students = []
        for face in faces:
            if face.identity != "Unknown" and face.confidence > threshold:
                # Try to extract the last 3 digits from identity string
                try:
                    if face.identity.isdigit():
                        # If identity is all digits, take last 3
                        detected_students.append(face.identity[-3:])
                    else:
                        # Otherwise try to find digits in the string
                        digits = ''.join(filter(str.isdigit, face.identity))
                        if len(digits) >= 3:
                            detected_students.append(digits[-3:])
                except:
                    # If there's any error in parsing, just continue
                    pass
        
        # Return the response with detected student IDs
        return {
            "image_base64": img_base64,
            "faces": [face.dict() for face in faces],
            "status": "success",
            "message": f"Recognized {len(faces)} faces",
            "detected_students": detected_students
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
