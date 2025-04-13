import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, Check, X, Edit2, ChevronLeft, ChevronRight } from 'lucide-react';
import axios from 'axios';

export default function AttendanceReview() {
  const location = useLocation();
  const navigate = useNavigate();
  const attendanceDataFromStorage = JSON.parse(sessionStorage.getItem('attendanceData') || '{}');
  const { attendanceData = attendanceDataFromStorage.attendance || [], images_base64 = attendanceDataFromStorage.images_base64 || [], formData = attendanceDataFromStorage || {} } = location.state || {};
  
  const [attendance, setAttendance] = useState(attendanceData);
  const [isEditing, setIsEditing] = useState(false);
  const [activeImageIndex, setActiveImageIndex] = useState(0);
  const [error, setError] = useState(null);

  console.log('AttendanceReview images_base64:', images_base64);
  console.log('AttendanceReview formData:', formData);

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

  const handleSaveAttendance = async () => {
    setError(null);

    // Validate formData
    const requiredFields = ['dept_name', 'year', 'section_name', 'subject_code', 'date', 'time'];
    const missingFields = requiredFields.filter(field => !formData[field] || formData[field] === '');
    if (missingFields.length > 0) {
      const errorMsg = `Missing required fields: ${missingFields.join(', ')}`;
      setError(errorMsg);
      alert(errorMsg);
      return;
    }

    // Validate year
    const year = parseInt(formData.year, 10);
    if (isNaN(year) || year <= 0) {
      const errorMsg = 'Year must be a valid positive number';
      setError(errorMsg);
      alert(errorMsg);
      return;
    }

    // Validate date format
    let dayOfWeek;
    try {
      const dateObj = new Date(formData.date);
      dayOfWeek = dateObj.toLocaleString('en-US', { weekday: 'long' });
      if (isNaN(dateObj.getTime())) {
        throw new Error('Invalid date');
      }
    } catch (e) {
      const errorMsg = 'Invalid date format. Use MM/DD/YYYY';
      setError(errorMsg);
      alert(errorMsg);
      return;
    }

    // Validate attendance
    if (!attendance.length) {
      const errorMsg = 'No attendance data to submit';
      setError(errorMsg);
      alert(errorMsg);
      return;
    }
    for (const entry of attendance) {
      if (!entry.register_number || !entry.name || typeof entry.is_present !== 'number') {
        const errorMsg = `Invalid attendance entry for ${entry.register_number || 'unknown'}`;
        setError(errorMsg);
        alert(errorMsg);
        return;
      }
    }

    let normalizedTime = formData.time;
    try {
      const timeObj = new Date(`1970-01-01T${formData.time}`);
      normalizedTime = timeObj.toTimeString().slice(0, 5);
    } catch (e) {
      console.warn('Invalid time format, using raw:', formData.time);
    }

    const formDataToSend = new FormData();
    formDataToSend.append('dept_name', formData.dept_name);
    formDataToSend.append('year', String(year));
    formDataToSend.append('section_name', formData.section_name);
    formDataToSend.append('subject_code', formData.subject_code);
    formDataToSend.append('date', formData.date);
    formDataToSend.append('time', normalizedTime);
    const attendanceData = attendance.map(entry => ({
      register_number: entry.register_number,
      name: entry.name,
      is_present: entry.is_present
    }));
    formDataToSend.append('attendance', JSON.stringify(attendanceData));

    console.log('Submitting attendance:', {
      dept_name: formData.dept_name,
      year,
      section_name: formData.section_name,
      subject_code: formData.subject_code,
      date: formData.date,
      time: normalizedTime,
      dayOfWeek,
      attendance: attendanceData,
    });

    try {
      await axios.post('http://localhost:8000/submit-attendance', formDataToSend);
      alert('Attendance saved successfully!');
      setIsEditing(false);
      sessionStorage.removeItem('attendanceForm');
      sessionStorage.removeItem('attendanceData');
      navigate('/attendance-assist');
    } catch (error) {
      console.error('Error saving attendance:', error);
      const errorMsg = error.response?.status === 404
        ? `Schedule error: ${error.response.data.detail}. Please check the timetable for ${dayOfWeek} at ${normalizedTime}.`
        : `Failed to save attendance: ${error.response?.data?.detail || error.message}`;
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
    <div className="min-h-screen bg-gray-100">
      <header className="border-b border-gray-200 p-4 flex justify-between items-center bg-white shadow-sm">
        <h1 className="text-xl font-semibold text-gray-800">Attendance Review</h1>
        <button
          onClick={() => navigate('/attendance-assist')}
          className="p-2 text-gray-600 hover:text-gray-800 flex items-center gap-2"
        >
          <ArrowLeft size={24} />
          Back
        </button>
      </header>

      <main className="p-6 max-w-2xl mx-auto">
        {error && <p className="text-red-500 text-center mb-4">{error}</p>}
        {attendance.length === 0 ? (
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <p className="text-gray-600 text-lg font-medium">No attendance data available</p>
          </div>
        ) : (
          <div className="space-y-6">
            {images_base64.length > 0 ? (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-lg font-semibold text-gray-800 mb-4">Processed Images</h2>
                <div className="flex justify-between items-center mb-6">
                  <button
                    onClick={() => goToImage(activeImageIndex - 1)}
                    className="p-2 text-gray-600 bg-gray-200 rounded-full hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition"
                    disabled={images_base64.length <= 1}
                  >
                    <ChevronLeft size={20} />
                  </button>
                  <span className="text-sm text-gray-600 font-medium">
                    {activeImageIndex + 1} / {images_base64.length}
                  </span>
                  <button
                    onClick={() => goToImage(activeImageIndex + 1)}
                    className="p-2 text-gray-600 bg-gray-200 rounded-full hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition"
                    disabled={images_base64.length <= 1}
                  >
                    <ChevronRight size={20} />
                  </button>
                </div>
                <div className="relative flex justify-center mb-4">
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
                      (activeImageIndex === 0 && index === images_base64.length - 1)
                    ) {
                      positionClass = "absolute left-4 z-20";
                      rotateStyle = { transform: "rotate(-6deg)" };
                    } else if (
                      index === activeImageIndex + 1 ||
                      (activeImageIndex === images_base64.length - 1 && index === 0)
                    ) {
                      positionClass = "absolute right-4 z-20";
                      rotateStyle = { transform: "rotate(6deg)" };
                    }

                    return (
                      <div
                        key={index}
                        className={`w-72 h-72 border border-gray-300 shadow-lg rounded-lg overflow-hidden transition-all duration-300 bg-white ${positionClass}`}
                        style={rotateStyle}
                      >
                        <img
                          src={`data:image/jpeg;base64,${image}`}
                          alt={`Processed image ${index + 1}`}
                          className="w-full h-full object-cover"
                          onError={() => console.error(`Failed to load image ${index}`)}
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-lg p-6 text-center">
                <p className="text-gray-600 text-lg font-medium">No processed images available</p>
              </div>
            )}

            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold text-gray-800">Detected Students</h2>
                {!isEditing && (
                  <button
                    onClick={handleEditToggle}
                    className="flex items-center gap-2 px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition"
                  >
                    <Edit2 size={16} />
                    Edit Attendance
                  </button>
                )}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left rounded-lg">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="py-3 px-4 text-gray-700 font-semibold rounded-tl-lg">Register Number</th>
                      <th className="py-3 px-4 text-gray-700 font-semibold">Name</th>
                      <th className="py-3 px-4 text-gray-700 font-semibold rounded-tr-lg">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {attendance.map((student) => (
                      <tr key={student.register_number} className="border-b last:border-0 hover:bg-gray-50">
                        <td className="py-3 px-4">{student.register_number}</td>
                        <td className="py-3 px-4">{student.name}</td>
                        <td className="py-3 px-4">
                          <button
                            onClick={() => handleTogglePresence(student.register_number)}
                            className={`flex items-center gap-1 ${isEditing ? 'cursor-pointer' : 'cursor-default'}`}
                            disabled={!isEditing}
                          >
                            {student.is_present ? (
                              <span className="text-green-600 flex items-center gap-1">
                                <Check size={16} /> Present
                              </span>
                            ) : (
                              <span className="text-red-600 flex items-center gap-1">
                                <X size={16} /> Absent
                              </span>
                            )}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 text-gray-700">
                <p>
                  Total Students: {attendance.length} | Present:{' '}
                  {attendance.filter((s) => s.is_present).length} | Absent:{' '}
                  {attendance.filter((s) => !s.is_present).length}
                </p>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-6 flex justify-end gap-4">
              {isEditing ? (
                <>
                  <button
                    onClick={handleCancelEdit}
                    className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSaveAttendance}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                  >
                    Save Changes
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => navigate('/attendance-assist')}
                    className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
                  >
                    Back to Upload
                  </button>
                  <button
                    onClick={handleSaveAttendance}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                  >
                    Save Attendance
                  </button>
                </>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

