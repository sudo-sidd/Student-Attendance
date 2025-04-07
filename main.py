from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

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

@app.get('/sections')
async def get_sections():
    res = data.groupby("Section")
    return {
        "sections": list(res.groups.keys())
    }

@app.get('/class/{section}')
async def get_class(section :str):
    filtered_data = data[data["Section"] == section]
    return JSONResponse(content=filtered_data.to_dict(orient="records"))
    

def process_image(path):
    global SELECTED_CLASS # based on this filter the data 
    pass


@app.post("/upload-image/")
async def upload_image(image: UploadFile = File(...)):
    save_path = Path("uploads") / "temp.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return {"message": f"Saved to {save_path}"}