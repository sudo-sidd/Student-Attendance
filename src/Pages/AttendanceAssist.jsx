import { useState, useRef } from 'react';
import { Menu, Plus, ChevronLeft, ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function AttendanceAssist() {
  const [images, setImages] = useState([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleImageUpload = (event) => {
    const files = Array.from(event.target.files);
    const newImages = files.map((file) => ({
      id: Date.now() + Math.random(),
      url: URL.createObjectURL(file),
      name: file.name,
      file,
    }));
    setImages((prev) => [...prev, ...newImages]);
    setError(null);
  };

  const goToImage = (index) => {
    if (images.length <= 1) return;
    if (index < 0) {
      setActiveIndex(images.length - 1);
    } else if (index >= images.length) {
      setActiveIndex(0);
    } else {
      setActiveIndex(index);
    }
  };

  const handleProcessImages = async () => {
    if (images.length === 0) {
      setError('Please upload at least one image');
      return;
    }

    setIsProcessing(true);
    const formData = new FormData();
    images.forEach((image) => formData.append('images', image.file));

    const attendanceForm = JSON.parse(sessionStorage.getItem('attendanceForm') || '{}');
    console.log('Attendance form data:', attendanceForm);
    const requiredFields = ['dept_name', 'year', 'section_name', 'subject_code', 'date', 'time'];
    const missingFields = requiredFields.filter((field) => !attendanceForm[field]);
    if (missingFields.length > 0) {
      setError(`Missing required fields: ${missingFields.join(', ')}`);
      setIsProcessing(false);
      return;
    }

    // Normalize time to HH:MM
    let normalizedTime = attendanceForm.time;
    try {
      const timeObj = new Date(`1970-01-01T${attendanceForm.time}`);
      normalizedTime = timeObj.toTimeString().slice(0, 5); // e.g., "11:48"
    } catch (e) {
      console.warn('Invalid time format, using raw:', attendanceForm.time);
    }

    formData.append('dept_name', attendanceForm.dept_name);
    formData.append('year', attendanceForm.year);
    formData.append('section_name', attendanceForm.section_name);
    formData.append('subject_code', attendanceForm.subject_code);
    formData.append('date', attendanceForm.date);
    formData.append('time', normalizedTime);
    formData.append('threshold', '0.45');

    try {
      const response = await axios.post('http://localhost:8000/process-images', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      console.log('Backend response:', response.data);

      const imagesBase64 = response.data.images_base64 || [];

      // Store only essential data without the large base64 images
      sessionStorage.setItem('attendanceData', JSON.stringify({
        ...attendanceForm,
        time: normalizedTime,
        attendance: response.data.attendance,
        // Removed images_base64 to prevent QuotaExceededError
      }));

      navigate('/review', {
        state: {
          attendanceData: response.data.attendance || [],
          images_base64: imagesBase64,
          formData: { ...attendanceForm, time: normalizedTime },
        },
      });
    } catch (error) {
      console.error('Error processing images:', error);
      let errorMessage = 'Failed to process images';
      if (error.response) {
        if (error.response.status === 404) {
          errorMessage = 'Section not found. Check department, year, or section name.';
        } else {
          errorMessage = error.response.data?.detail || error.message;
        }
      }
      setError(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="border-b border-gray-200 p-4 flex justify-between items-center bg-white shadow-sm">
        <h1 className="text-xl font-semibold text-gray-800">AI Attendance Assist</h1>
      </header>

      <main className="p-6 max-w-2xl mx-auto">
        {error && <p className="text-red-500 text-center mb-4">{error}</p>}
        {images.length === 0 ? (
          <div className="mt-12">
            <div className="relative flex justify-center mb-8">
              <div className="w-32 h-40 border border-gray-300 bg-gray-50 absolute left-8 transform -rotate-6 shadow-md"></div>
              <div className="w-48 h-48 border-2 border-gray-400 border-dashed flex flex-col items-center justify-center p-4 z-10 bg-white rounded-lg shadow-lg">
                <div
                  onClick={() => fileInputRef.current.click()}
                  className="w-16 h-16 rounded-full border-2 border-gray-300 flex items-center justify-center mb-4 cursor-pointer hover:bg-gray-100 transition"
                >
                  <Plus size={24} className="text-gray-500" />
                </div>
                <p className="text-sm text-center text-gray-600 font-medium">Upload 1 or more photos of the class</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept="image/*"
                  className="hidden"
                  onChange={handleImageUpload}
                />
              </div>
              <div className="w-32 h-40 border border-gray-300 bg-gray-50 absolute right-8 transform rotate-6 shadow-md"></div>
            </div>
            <p className="text-center text-gray-600 text-lg font-medium">Waiting for upload</p>
          </div>
        ) : (
          <div className="mt-6">
            <div className="flex justify-between items-center mb-6">
              <button
                onClick={() => goToImage(activeIndex - 1)}
                className="p-2 text-gray-600 bg-gray-200 rounded-full hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition"
                disabled={images.length <= 1}
              >
                <ChevronLeft size={20} />
              </button>
              <span className="text-sm text-gray-600 font-medium">
                {activeIndex + 1} / {images.length}
              </span>
              <button
                onClick={() => goToImage(activeIndex + 1)}
                className="p-2 text-gray-600 bg-gray-200 rounded-full hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition"
                disabled={images.length <= 1}
              >
                <ChevronRight size={20} />
              </button>
            </div>

            <div className="relative flex justify-center mb-8">
              {images.map((image, index) => {
                let positionClass = "hidden";
                let rotateStyle = {};
                if (index === activeIndex) {
                  positionClass = "z-30";
                  rotateStyle = {};
                } else if (index === activeIndex - 1 || (activeIndex === 0 && index === images.length - 1)) {
                  positionClass = "absolute left-4 z-20";
                  rotateStyle = { transform: "rotate(-6deg)" };
                } else if (index === activeIndex + 1 || (activeIndex === images.length - 1 && index === 0)) {
                  positionClass = "absolute right-4 z-20";
                  rotateStyle = { transform: "rotate(6deg)" };
                }

                return (
                  <div
                    key={image.id}
                    className={`w-72 h-72 border border-gray-300 shadow-lg rounded-lg ${positionClass}`}
                    style={rotateStyle}>
                    <img src={image.url} alt={image.name} className="w-full h-full object-cover" />
                  </div>
                );
              })}
            </div>

            <div className="flex justify-center gap-4">
              <button
                onClick={() => fileInputRef.current.click()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
              >
                Upload More
              </button>
              <button
                onClick={handleProcessImages}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 transition flex items-center gap-2"
                disabled={isProcessing}
              >
                {isProcessing ? (
                  <>
                    <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8v8h8a8 8 0 01-16 0z"
                      />
                    </svg>
                    Processing...
                  </>
                ) : (
                  'Process Images'
                )}
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}