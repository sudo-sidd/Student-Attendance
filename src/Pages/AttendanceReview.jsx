import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Check,
  X,
  Edit2,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import axios from "axios";
import Header from "../components/Header";
import Footer from "../components/Footer";

export default function AttendanceReview() {
  const location = useLocation();
  const navigate = useNavigate();
  const attendanceDataFromStorage = JSON.parse(
    sessionStorage.getItem("attendanceData") || "{}"
  );
  const {
    attendanceData = attendanceDataFromStorage.attendance || [],
    images_base64 = attendanceDataFromStorage.images_base64 || [],
    formData = attendanceDataFromStorage || {},
  } = location.state || {};

  const [attendance, setAttendance] = useState(attendanceData);
  const [isEditing, setIsEditing] = useState(false);
  const [activeImageIndex, setActiveImageIndex] = useState(0);
  const [error, setError] = useState(null);

  const [previewImage, setPreviewImage] = useState(null);

  console.log("AttendanceReview images_base64:", images_base64);
  console.log("AttendanceReview formData:", formData);

  const handleTogglePresence = (register_number) => {
    if (!isEditing) return;
    setAttendance((prev) =>
      prev.map((student) =>
        student.register_number === register_number
          ? { ...student, is_present: student.is_present ? 0 : 1 }
          : student
      )
    );
  };

  // const handleSaveAttendance = async () => {
  //   setError(null);

  //   // Validate formData
  //   const requiredFields = ['dept_name', 'year', 'section_name', 'subject_code', 'date', 'time'];
  //   const missingFields = requiredFields.filter(field => !formData[field] || formData[field] === '');
  //   if (missingFields.length > 0) {
  //     const errorMsg = `Missing required fields: ${missingFields.join(', ')}`;
  //     setError(errorMsg);
  //     alert(errorMsg);
  //     return;
  //   }

  //   // Validate year
  //   const year = parseInt(formData.year, 10);
  //   if (isNaN(year) || year <= 0) {
  //     const errorMsg = 'Year must be a valid positive number';
  //     setError(errorMsg);
  //     alert(errorMsg);
  //     return;
  //   }

  //   // Validate date format
  //   let dayOfWeek;
  //   try {
  //     const dateObj = new Date(formData.date);
  //     dayOfWeek = dateObj.toLocaleString('en-US', { weekday: 'long' });
  //     if (isNaN(dateObj.getTime())) {
  //       throw new Error('Invalid date');
  //     }
  //   } catch (e) {
  //     const errorMsg = 'Invalid date format. Use MM/DD/YYYY';
  //     setError(errorMsg);
  //     alert(errorMsg);
  //     return;
  //   }

  //   // Validate attendance
  //   if (!attendance.length) {
  //     const errorMsg = 'No attendance data to submit';
  //     setError(errorMsg);
  //     alert(errorMsg);
  //     return;
  //   }
  //   for (const entry of attendance) {
  //     if (!entry.register_number || !entry.name || typeof entry.is_present !== 'number') {
  //       const errorMsg = `Invalid attendance entry for ${entry.register_number || 'unknown'}`;
  //       setError(errorMsg);
  //       alert(errorMsg);
  //       return;
  //     }
  //   }

  //   let normalizedTime = formData.time;
  //   try {
  //     const timeObj = new Date(`1970-01-01T${formData.time}`);
  //     normalizedTime = timeObj.toTimeString().slice(0, 5);
  //   } catch (e) {
  //     console.warn('Invalid time format, using raw:', formData.time);
  //   }

  //   const formDataToSend = new FormData();
  //   formDataToSend.append('dept_name', formData.dept_name);
  //   formDataToSend.append('year', String(year));
  //   formDataToSend.append('section_name', formData.section_name);
  //   formDataToSend.append('subject_code', formData.subject_code);
  //   formDataToSend.append('date', formData.date);
  //   formDataToSend.append('time', normalizedTime);
  //   const attendanceData = attendance.map(entry => ({
  //     register_number: entry.register_number,
  //     name: entry.name,
  //     is_present: entry.is_present
  //   }));
  //   formDataToSend.append('attendance', JSON.stringify(attendanceData));

  //   console.log('Submitting attendance:', {
  //     dept_name: formData.dept_name,
  //     year,
  //     section_name: formData.section_name,
  //     subject_code: formData.subject_code,
  //     date: formData.date,
  //     time: normalizedTime,
  //     dayOfWeek,
  //     attendance: attendanceData,
  //   });

  //   try {
  //     await axios.post('http://localhost:8000/submit-attendance', formDataToSend);
  //     alert('Attendance saved successfully!');
  //     setIsEditing(false);
  //     sessionStorage.removeItem('attendanceForm');
  //     sessionStorage.removeItem('attendanceData');
  //     navigate('/attendance-assist');
  //   } catch (error) {
  //     console.error('Error saving attendance:', error);
  //     const errorMsg = error.response?.status === 404
  //       ? `Schedule error: ${error.response.data.detail}. Please check the timetable for ${dayOfWeek} at ${normalizedTime}.`
  //       : `Failed to save attendance: ${error.response?.data?.detail || error.message}`;
  //     setError(errorMsg);
  //     alert(errorMsg);
  //   }
  // };

  const handleSaveAttendance = async () => {
    setError(null);

    // Validate attendance
    if (!attendance.length) {
      const errorMsg = "No attendance data to submit";
      setError(errorMsg);
      alert(errorMsg);
      return;
    }
    for (const entry of attendance) {
      if (
        !entry.register_number ||
        !entry.name ||
        typeof entry.is_present !== "number"
      ) {
        const errorMsg = `Invalid attendance entry for ${
          entry.register_number || "unknown"
        }`;
        setError(errorMsg);
        alert(errorMsg);
        return;
      }
    }

    const formDataToSend = new FormData();
    formDataToSend.append("timetable_id", formData.timetable_id);
    const attendanceData = attendance.map((entry) => ({
      register_number: entry.register_number,
      name: entry.name,
      is_present: entry.is_present,
    }));
    formDataToSend.append("attendance", JSON.stringify(attendanceData));

    try {
      await axios.post(
        "http://localhost:8000/submit-attendance",
        formDataToSend
      );
      alert("Attendance saved successfully!");
      setIsEditing(false);
      sessionStorage.removeItem("attendanceForm");
      sessionStorage.removeItem("attendanceData");
      navigate("/attendance-assist");
    } catch (error) {
      console.error("Error saving attendance:", error);
      const errorMsg =
        error.response?.data?.detail || "Failed to save attendance";
      setError(errorMsg);
      alert(errorMsg);
    }
  };

  const handleEditToggle = () => {
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setAttendance(attendanceData);
  };

  const goToImage = (index) => {
    if (images_base64.length <= 1) return;
    if (index < 0) {
      setActiveImageIndex(images_base64.length - 1);
    } else if (index >= images_base64.length) {
      setActiveImageIndex(0);
    } else {
      setActiveImageIndex(index);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
    <Header />

    <main className="mb-12 p-6 max-w-4xl mx-auto mt-24">
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded-md">
          <p className="text-red-700">{error}</p>
        </div>
      )}
      
      {attendance.length === 0 ? (
        <div className="bg-white rounded-lg shadow-md p-8 text-center">
          <p className="text-gray-600 text-lg">No attendance data available</p>
        </div>
      ) : (
        <div className="space-y-8">
          {/* Images Section */}
          {images_base64.length > 0 ? (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 border-b pb-2">
                Processed Images
              </h2>
              <div className="flex justify-between items-center mb-6">
                <button
                  onClick={() => goToImage(activeImageIndex - 1)}
                  className="p-2.5 text-gray-700 bg-gray-100 rounded-full hover:bg-gray-200 disabled:opacity-40 disabled:cursor-not-allowed transition"
                  disabled={images_base64.length <= 1}
                >
                  <ChevronLeft size={20} />
                </button>
                <span className="text-sm text-gray-600 font-medium bg-gray-100 px-3 py-1 rounded-full">
                  {activeImageIndex + 1} / {images_base64.length}
                </span>
                <button
                  onClick={() => goToImage(activeImageIndex + 1)}
                  className="p-2.5 text-gray-700 bg-gray-100 rounded-full hover:bg-gray-200 disabled:opacity-40 disabled:cursor-not-allowed transition"
                  disabled={images_base64.length <= 1}
                >
                  <ChevronRight size={20} />
                </button>
              </div>
              <div className="relative flex justify-center mb-4 h-80">
                {images_base64.map((image, index) => {
                  if (!image) {
                    console.warn(`Invalid base64 image at index ${index}`);
                    return null;
                  }
                  let positionClass = "hidden";
                  let rotateStyle = {};
                  if (index === activeImageIndex) {
                    positionClass = "z-30";
                    rotateStyle = {};
                  } else if (
                    index === activeImageIndex - 1 ||
                    (activeImageIndex === 0 &&
                      index === images_base64.length - 1)
                  ) {
                    positionClass = "absolute left-4 z-20";
                    rotateStyle = { transform: "rotate(-6deg)" };
                  } else if (
                    index === activeImageIndex + 1 ||
                    (activeImageIndex === images_base64.length - 1 &&
                      index === 0)
                  ) {
                    positionClass = "absolute right-4 z-20";
                    rotateStyle = { transform: "rotate(6deg)" };
                  }

                  return (
                    <div
                      key={index}
                      className={`w-80 h-72 border border-gray-200 shadow-lg rounded-lg overflow-hidden transition-all duration-300 bg-white ${positionClass}`}
                      style={rotateStyle}
                    >
                      <img
                        src={`data:image/jpeg;base64,${image}`}
                        alt={`Processed image ${index + 1}`}
                        className="w-full h-full object-cover cursor-pointer hover:opacity-90 transition"
                        onClick={() => setPreviewImage(image)}
                        onError={() =>
                          console.error(`Failed to load image ${index}`)
                        }
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-md p-6 text-center">
              <p className="text-gray-600">
                No processed images available
              </p>
            </div>
          )}

          {/* Attendance Section */}
                <div className="bg-white rounded-lg shadow-md overflow-hidden">
                <div className="p-6 border-b border-gray-100 flex justify-between items-center">
                  <h2 className="text-xl font-semibold text-gray-800">
                  Attendance Record
                  </h2>
                  {!isEditing && (
                  <button
                    onClick={handleEditToggle}
                    className="flex items-center gap-2 px-4 py-2.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition shadow-sm"
                  >
                    <Edit2 size={16} />
                    Edit Attendance
                  </button>
                  )}
                </div>
                
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                    <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Date
                    </th>
                    <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Register Number
                    </th>
                    <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Name
                    </th>
                    <th className="px-6 py-3.5 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Status
                    </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {attendance.map((student) => (
                    <tr
                      key={student.register_number}
                      className="hover:bg-gray-50 transition-colors"
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                      {new Date().toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800">
                      {student.register_number}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                      {student.name}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-center">
                      <button
                        onClick={() =>
                        handleTogglePresence(student.register_number)
                        }
                        className={`inline-flex items-center justify-center rounded-md px-3 py-1.5 transition-colors ${
                        isEditing 
                          ? "cursor-pointer" 
                          : "cursor-default"
                        } ${
                        student.is_present 
                          ? "bg-green-50 text-green-700 hover:bg-green-100 border border-green-200" 
                          : "bg-red-50 text-red-700 hover:bg-red-100 border border-red-200"
                        }`}
                        disabled={!isEditing}
                      >
                        {student.is_present ? (
                        <span className="flex items-center gap-1.5">
                          <Check size={16} strokeWidth={2.5} /> Present
                        </span>
                        ) : (
                        <span className="flex items-center gap-1.5">
                          <X size={16} strokeWidth={2.5} /> Absent
                        </span>
                        )}
                      </button>
                      </td>
                    </tr>
                    ))}
                  </tbody>
                  </table>
                </div>

                {/* Stats Summary */}
            <div className="bg-gray-50 px-6 py-3 border-t border-gray-200">
              <div className="flex justify-between items-center">
                <div className="flex space-x-6">
                  <div className="text-sm">
                    <span className="text-gray-500">Total: </span>
                    <span className="font-medium text-gray-900">{attendance.length}</span>
                  </div>
                  <div className="text-sm">
                    <span className="text-gray-500">Present: </span>
                    <span className="font-medium text-green-600">{attendance.filter(s => s.is_present).length}</span>
                  </div>
                  <div className="text-sm">
                    <span className="text-gray-500">Absent: </span>
                    <span className="font-medium text-red-600">{attendance.filter(s => !s.is_present).length}</span>
                  </div>
                </div>
              </div>
            </div>
               
          {/* Action Buttons */}
          <div className="bg-white rounded-lg shadow-md p-6 flex justify-end gap-3">
            {isEditing ? (
              <>
                <button
                  onClick={handleCancelEdit}
                  className="px-5 py-2.5 bg-white text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50 transition"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveAttendance}
                  className="px-5 py-2.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 shadow-sm transition"
                >
                  Save Changes
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={() => navigate("/attendance-assist")}
                  className="px-5 py-2.5 flex items-center gap-2 bg-white text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50 transition"
                >
                  <ArrowLeft size={16} />
                  Back
                </button>
                <button
                  onClick={handleSaveAttendance}
                  className="px-5 py-2.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 shadow-sm transition"
                >
                  Save Attendance
                </button>
              </>
            )}
          </div>
          </div>
         
        
            
        
      </div>
      )}
        </main>
    {/* Image Preview Modal */}
    {previewImage && (
      <div
        className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
        onClick={() => setPreviewImage(null)}
      >
        <div className="relative max-w-4xl w-full">
          <img
            src={`data:image/jpeg;base64,${previewImage}`}
            alt="Preview"
            className="max-w-full max-h-[85vh] object-contain mx-auto rounded-lg shadow-2xl"
          />
          <button 
            className="absolute top-2 right-2 bg-black bg-opacity-50 text-white rounded-full p-1 hover:bg-opacity-70 transition"
            onClick={(e) => {
              e.stopPropagation();
              setPreviewImage(null);
            }}
          >
            <X size={20} />
          </button>
        </div>
      </div>
    )}
    
    <Footer />
  </div>
  );
}
