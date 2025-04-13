from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sqlite3
from pathlib import Path
import logging
from typing import List
from datetime import datetime
import pandas as pd
import io
import torch
import cv2
import numpy as np
import base64
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
from ultralytics import YOLO

# Assuming LightCNN is in a local module or submodule
from backend.LightCNN.light_cnn import LightCNN_29Layers_v2  # Adjust path as needed

app = FastAPI(title="AI Student Attendance System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Adjust to your React frontend port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "attendance.db"

# Global variables for models
face_model = None
yolo_model = None
gallery = None
device = None
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Pydantic Models
class DepartmentResponse(BaseModel):
    dept_name: str

class BatchResponse(BaseModel):
    batch_id: int
    dept_name: str
    year: int

class SectionResponse(BaseModel):
    section_id: int
    batch_id: int
    section_name: str

class StudentResponse(BaseModel):
    register_number: str
    name: str
    section_id: int
    section_name: str
    batch_id: int

class AttendanceEntry(BaseModel):
    register_number: str
    name: str
    is_present: int

class AttendanceRequest(BaseModel):
    dept_name: str
    year: int
    section_name: str
    date: str
    time: str
    attendance: List[AttendanceEntry]

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

# Database Connection
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Load gallery from database (simplified for SQLite)
def load_gallery_for_section(dept_name: str, year: int, section_name: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.register_number, s.embedding
            FROM Students s
            JOIN Sections sec ON s.section_id = sec.section_id
            JOIN Batches b ON sec.batch_id = b.batch_id
            JOIN Departments d ON b.dept_id = d.dept_id
            WHERE d.dept_name = ? AND b.year = ? AND sec.section_name = ?
            AND s.embedding IS NOT NULL
        """, (dept_name, year, section_name))
        gallery_data = cursor.fetchall()
        conn.close()

        gallery = {}
        for row in gallery_data:
            # Assuming embeddings are stored as BLOBs; convert to numpy array
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            gallery[row["register_number"]] = embedding
        print(f"✓ Gallery loaded for {dept_name}-{year}-{section_name} with {len(gallery)} identities")
        return gallery
    except Exception as e:
        print(f"❌ Error loading gallery: {e}")
        return {}

# Startup event to load models
@app.on_event("startup")
async def startup_event():
    global face_model, yolo_model, device
    
    if not Path(DB_PATH).exists():
        raise FileNotFoundError("Database file 'attendance.db' not found.")

    # Model paths (adjust as needed)
    model_path = "backend/checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar"
    yolo_path = "backend/yolo/weights/yolo11n-face.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for inference")

    # Load face recognition model
    face_model = LightCNN_29Layers_v2(num_classes=100)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        new_state_dict = {}
        state_dict = checkpoint.get("state_dict", checkpoint)
        for k, v in state_dict.items():
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

# Image processing function
def process_image(image_bytes, threshold=0.6, gallery=None):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    result_img = img.copy()
    faces = []
    face_id = 0

    # Detect faces using YOLO
    results = yolo_model(img)

    for result in results:
        for box in result.boxes:
            face_id += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Add padding
            h, w = img.shape[:2]
            face_w, face_h = x2 - x1, y2 - y1
            pad_x, pad_y = int(face_w * 0.2), int(face_h * 0.2)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            face = img[y1:y2, x1:x2]
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_face = clahe.apply(gray_face)
            enhanced_face = cv2.GaussianBlur(enhanced_face, (3, 3), 0)
            enhanced_face = cv2.normalize(enhanced_face, None, 0, 255, cv2.NORM_MINMAX)

            face_pil = Image.fromarray(enhanced_face)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                _, embedding = face_model(face_tensor)
                face_embedding = embedding.cpu().squeeze().numpy()

            matches = []
            for identity, gallery_embedding in gallery.items():
                similarity = 1 - cosine(face_embedding, gallery_embedding)
                if similarity > threshold:
                    matches.append((identity, similarity))

            matches.sort(key=lambda x: x[1], reverse=True)
            best_match = "Unknown"
            best_score = -1
            if matches:
                best_match, best_score = matches[0]

            # Draw on image
            color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            label = f"{best_match} ({best_score:.2f})" if best_match != "Unknown" else "Unknown"
            cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            faces.append(RecognitionResult(
                face_id=face_id,
                identity=best_match,
                confidence=float(best_score) if best_score > -1 else 0.0,
                bbox=[int(x1), int(y1), int(x2), int(y2)]
            ))

    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64, faces

# Existing Endpoints (abbreviated for brevity)
@app.get("/departments", response_model=List[DepartmentResponse])
async def get_departments():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT dept_name FROM Departments ORDER BY dept_name")
        departments = [{"dept_name": row["dept_name"]} for row in cursor.fetchall()]
        conn.close()
        return departments
    except Exception as e:
        logging.error(f"Error fetching departments: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/students", response_model=List[StudentResponse])
async def get_students():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT st.register_number, st.name, st.section_id, sec.section_name, sec.batch_id
            FROM Students st
            JOIN Sections sec ON st.section_id = sec.section_id
        """)
        students = [
            {
                "register_number": row["register_number"],
                "name": row["name"],
                "section_id": row["section_id"],
                "section_name": row["section_name"],
                "batch_id": row["batch_id"]
            }
            for row in cursor.fetchall()
        ]
        conn.close()
        return students
    except Exception as e:
        logging.error(f"Error fetching students: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/upload-students-csv", response_model=dict)
async def upload_students_csv(
    file: UploadFile = File(...),
    dept_name: str = Form(...),
    year: int = Form(...)
):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT dept_id FROM Departments WHERE dept_name = ?", (dept_name,))
        dept_result = cursor.fetchone()
        if not dept_result:
            conn.close()
            raise HTTPException(status_code=404, detail="Department not found")
        dept_id = dept_result["dept_id"]

        cursor.execute("SELECT batch_id FROM Batches WHERE dept_id = ? AND year = ?", (dept_id, year))
        batch_result = cursor.fetchone()
        if not batch_result:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Batch not found for {dept_name} year {year}")
        batch_id = batch_result["batch_id"]

        cursor.execute("SELECT section_id, section_name FROM Sections WHERE batch_id = ?", (batch_id,))
        sections = {row["section_name"]: row["section_id"] for row in cursor.fetchall()}
        if not sections:
            conn.close()
            raise HTTPException(status_code=404, detail=f"No sections found for {dept_name} year {year}")

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        required_columns = {'RegisterNumber', 'Name', 'Section'}
        if not required_columns.issubset(df.columns):
            conn.close()
            raise HTTPException(status_code=400, detail="CSV must contain RegisterNumber, Name, and Section columns")

        students_data = []
        for _, row in df.iterrows():
            register_number = str(row['RegisterNumber']).strip()
            name = str(row['Name']).strip()
            section_name = str(row['Section']).strip()
            if section_name not in sections:
                conn.close()
                raise HTTPException(status_code=400, detail=f"Section '{section_name}' not found")
            section_id = sections[section_name]
            students_data.append((register_number, name, section_id))

        cursor.executemany("""
            INSERT INTO Students (register_number, name, section_id)
            VALUES (?, ?, ?)
            ON CONFLICT(register_number) DO UPDATE SET
                name = excluded.name,
                section_id = excluded.section_id
        """, students_data)
        conn.commit()

        cursor.execute("""
            SELECT st.register_number, st.name, st.section_id, sec.section_name, sec.batch_id
            FROM Students st
            JOIN Sections sec ON st.section_id = sec.section_id
        """)
        students = [
            {
                "register_number": row["register_number"],
                "name": row["name"],
                "section_id": row["section_id"],
                "section_name": row["section_name"],
                "batch_id": row["batch_id"]
            }
            for row in cursor.fetchall()
        ]
        conn.close()
        return {"message": "Students uploaded successfully", "students": students}
    except Exception as e:
        logging.error(f"Error uploading students CSV: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/submit-attendance")
async def submit_attendance(request: AttendanceRequest):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.section_id 
            FROM Sections s 
            JOIN Batches b ON s.batch_id = b.batch_id 
            JOIN Departments d ON b.dept_id = d.dept_id 
            WHERE d.dept_name = ? AND b.year = ? AND s.section_name = ?
        """, (request.dept_name, request.year, request.section_name))
        section_result = cursor.fetchone()
        if not section_result:
            conn.close()
            raise HTTPException(status_code=404, detail="Section not found")
        section_id = section_result["section_id"]

        day_of_week = datetime.strptime(request.date, "%m/%d/%Y").strftime("%A")
        cursor.execute("""
            SELECT timetable_id 
            FROM Timetable 
            WHERE section_id = ? 
            AND day_of_week = ? 
            AND start_time = ?
        """, (section_id, day_of_week, request.time))
        timetable_result = cursor.fetchone()
        if not timetable_result:
            conn.close()
            raise HTTPException(status_code=404, detail=f"No timetable slot found for {request.time} on {day_of_week}")
        timetable_id = timetable_result["timetable_id"]

        attendance_data = []
        for entry in request.attendance:
            cursor.execute("SELECT student_id FROM Students WHERE register_number = ?", (entry.register_number,))
            student_result = cursor.fetchone()
            if not student_result:
                conn.close()
                raise HTTPException(status_code=404, detail=f"Student {entry.register_number} not found")
            student_id = student_result["student_id"]
            attendance_data.append((student_id, timetable_id, request.date, entry.is_present))

        cursor.executemany("""
            INSERT OR REPLACE INTO Attendance (student_id, timetable_id, date, is_present) 
            VALUES (?, ?, ?, ?)
        """, attendance_data)
        conn.commit()
        conn.close()
        return {"message": "Attendance submitted successfully"}
    except Exception as e:
        logging.error(f"Error submitting attendance: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# New Image Recognition Endpoint
@app.post("/process-images", response_model=dict)
async def process_images(
    images: List[UploadFile] = File(...),
    dept_name: str = Form(...),
    year: int = Form(...),
    section_name: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    threshold: float = Form(0.6)
):
    if face_model is None or yolo_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Please check server logs.")

    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Validate section
        cursor.execute("""
            SELECT s.section_id 
            FROM Sections s 
            JOIN Batches b ON s.batch_id = b.batch_id 
            JOIN Departments d ON b.dept_id = d.dept_id 
            WHERE d.dept_name = ? AND b.year = ? AND s.section_name = ?
        """, (dept_name, year, section_name))
        section_result = cursor.fetchone()
        if not section_result:
            conn.close()
            raise HTTPException(status_code=404, detail="Section not found")
        section_id = section_result["section_id"]

        # Load gallery for this section
        gallery = load_gallery_for_section(dept_name, year, section_name)
        if not gallery:
            conn.close()
            raise HTTPException(status_code=400, detail="No face embeddings found for this section")

        # Get all students in the section
        cursor.execute("""
            SELECT register_number, name 
            FROM Students 
            WHERE section_id = ?
        """, (section_id,))
        all_students = {row["register_number"]: row["name"] for row in cursor.fetchall()}

        # Process images
        detected_students = set()
        for image in images:
            contents = await image.read()
            img_base64, faces = process_image(contents, threshold, gallery)
            for face in faces:
                if face.identity != "Unknown" and face.confidence > threshold:
                    detected_students.add(face.identity)

        # Prepare attendance data
        attendance = [
            {
                "register_number": reg_num,
                "name": name,
                "is_present": 1 if reg_num in detected_students else 0
            }
            for reg_num, name in all_students.items()
        ]

        conn.close()
        return {
            "attendance": attendance,
            "image_base64": img_base64 if images else None,  # Return last processed image
            "status": "success",
            "message": f"Processed {len(images)} images, recognized {len(detected_students)} students"
        }
    except Exception as e:
        logging.error(f"Error processing images: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)