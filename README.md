# AI Student Attendance System

A modern student attendance tracking system that employs facial recognition technology to automate the attendance process in educational institutions. Built with a React frontend and FastAPI backend.

## Features

- **Facial Recognition**: Automatically detect and mark student attendance using AI facial recognition.
- **User Roles**: Different interfaces for regular users, administrators, and super administrators.
- **Live Processing**: Upload class images for immediate processing and attendance marking.
- **Attendance Records**: Store, retrieve, and analyze attendance data.
- **Reporting**: Generate detailed attendance reports and export to Excel format.
- **User Management**: Admin panel for managing departments, batches, sections, subjects, and students.
- **CSV Import**: Bulk import student data via CSV files.

## Technology Stack

### Frontend

- React
- React Router DOM
- Tailwind CSS
- Axios (for API requests)

### Backend

- Python 3
- FastAPI
- SQLite
- PyTorch (for AI models)
- OpenCV
- YOLO (You Only Look Once) for face detection
- LightCNN for face recognition

## Architecture

The system follows a client-server architecture:

1. **Frontend**: React-based SPA with different views for attendance capturing, review, administration, and reporting.
2. **Backend**: FastAPI server that handles database operations, business logic, and hosts the AI models.
3. **AI Components**: Pre-trained models for face detection (YOLO) and recognition (LightCNN).
4. **Database**: SQLite database for storing attendance records and system configuration.

## Installation

### Prerequisites

- Node.js (v14+)
- Python (v3.9+)
- pip

### Backend Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Student-Attendance.git
cd Student-Attendance
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

4. Download the required AI models:

   - [LightCNN_29Layers_V2_checkpoint.pth.tar](https://github.com/AlfredXiangWu/LightCNN) and place in `backend/checkpoints/`
   - [yolo11n-face.pt](https://github.com/ultralytics/yolov5) and place in `backend/yolo/weights/`
5. Start the FastAPI server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. Navigate to the project directory:

```bash
cd Student-Attendance
```

2. Install the required Node.js packages:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173`

## Usage

### Student Attendance Form

1. Access the home page (`/`)
2. Select department, year, section, subject, and date
3. Click "Take Attendance" to proceed to the attendance assistance page

### Attendance Assistance

1. Upload images of the classroom with students
2. The system will process images using facial recognition
3. Review the auto-detected students

### Attendance Review

1. View automatically marked attendance
2. Edit attendance if needed
3. Save the attendance records

### Admin Panel

1. Access the admin page (`/admin`)
2. View and filter attendance reports
3. Generate and export reports

### Super Admin Panel

1. Access the super admin page (`/superadmin`)
2. Manage departments, batches, sections, subjects, and students
3. Configure time blocks for attendance periods

## API Reference

The system provides a FastAPI backend with API documentation available at `http://localhost:8000/docs` when the server is running.

Key API endpoints:

- `/process-images`: Process uploaded classroom images for attendance
- `/submit-attendance`: Save attendance records
- `/get-attendance`: Retrieve attendance records with filtering
- Various CRUD endpoints for managing departments, batches, sections, subjects, students, and time blocks

## Database Structure

The system uses SQLite with the following main tables:

- Departments
- Batches
- Sections
- Subjects
- Students
- Timetable
- Attendance
- TimeBlocks

## Development

### Project Structure

```
Student-Attendance/
├── src/                  # Frontend React code
│   ├── components/       # Reusable React components
│   ├── Pages/            # Main page components
│   └── App2.jsx          # Main application component
├── backend/              # Backend Python code
│   ├── LightCNN/         # Face recognition model
│   ├── yolo/             # Face detection model
│   └── face_recognition_api.py  # API for face recognition
├── main.py               # FastAPI application
└── attendance.db         # SQLite database
```

## Credits

- AI Face Recognition components use [LightCNN](https://github.com/AlfredXiangWu/LightCNN)
- Face detection powered by [YOLOv5](https://github.com/ultralytics/yolov5)
- Designed and developed by AIML Students
