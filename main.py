from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # allow your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = pd.read_csv('NAME_LIST.csv')

SELECTED_CLASS = "A"

# Models for request validation
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

def process_image(path):
    pass

@app.post("/upload-image/")
async def upload_image(image: UploadFile = File(...)):
    save_path = Path("uploads") / "temp.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return {"message": f"Saved to {save_path}"}

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