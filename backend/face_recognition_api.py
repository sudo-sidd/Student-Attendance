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

data = pd.read_csv('/mnt/sda1/Project/Student-Attendance/NAME_LIST.csv')

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

@app.get('/class/{section}')
async def get_class(section: str):
    filtered_data = data[data["Section"] == section]
    return JSONResponse(content=filtered_data.to_dict(orient="records"))

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

@app.on_event("startup") 
async def startup_event():
    """Load models and gallery at startup"""
    global face_model, yolo_model, gallery, device
    
    # Path configurations (customize these)
    model_path = "./checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar"
    gallery_path = "gal_3.pth"
    yolo_path = "./yolo/weights/yolo11n-face.pt"
    
    # Device setup
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
    
    # Load gallery
    try:
        gallery = torch.load(gallery_path)
        print(f"✓ Gallery loaded with {len(gallery)} identities")
    except Exception as e:
        print(f"❌ Error loading gallery: {e}")
        raise RuntimeError(f"Failed to load gallery: {e}")

def process_image(image_bytes, threshold=0.6):
    """
    Process an image for face recognition
    
    Args:
        image_bytes: Raw image bytes
        threshold: Recognition confidence threshold
        
    Returns:
        Tuple of (processed_image, recognition_results)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Process each face
    result_img = img.copy()
    faces = []
    face_id = 0
    
    # Detect faces using YOLO
    results = yolo_model(img)
    
    # Process each detected face
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Add padding around face
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
            
            # Convert BGR to grayscale PIL image
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
            
            # Get face tensor and extract embedding
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, embedding = face_model(face_tensor)
                face_embedding = embedding.cpu().squeeze().numpy()
            
            # Find best match
            best_match = "Unknown"
            best_score = -1
            
            for identity, gallery_embedding in gallery.items():
                # Calculate cosine similarity
                similarity = 1 - cosine(face_embedding, gallery_embedding)
                
                if similarity > threshold and similarity > best_score:
                    best_score = similarity
                    best_match = identity
            
            # Draw result on image
            if best_match != "Unknown":
                # Known identity - green box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{best_match} ({best_score:.2f})"
                cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Unknown - red box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(result_img, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add to results
            face_id += 1
            confidence = float(best_score) if best_score > 0 else 0.0
            faces.append(RecognitionResult(
                face_id=face_id,
                identity=best_match,
                confidence=confidence,
                bbox=[int(x1), int(y1), int(x2), int(y2)]
            ))
    
    # Encode image to base64 to return in response
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64, faces

@app.post("/upload-image/")
async def upload_and_recognize(
    file: UploadFile = File(...),
    threshold: float = Form(0.6)
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
