import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Calendar, Clock, FileText, ArrowRight, ArrowLeft } from "lucide-react";
import axios from "axios";

const AttendanceForm = () => {
  const [departments, setDepartments] = useState([]);
  const [years, setYears] = useState([]);
  const [sections, setSections] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [formData, setFormData] = useState({
    dept_name: "",
    year: "",
    section_name: "",
    subject_code: "",
    date: "",
    start_time: "",
    end_time: "",
    day_of_week: "",
  });
  const [error, setError] = useState(null);
  const [currentSlide, setCurrentSlide] = useState(1); // Track slide (1 or 2)
  const navigate = useNavigate();

  // Set current date and time on mount
  useEffect(() => {
    const now = new Date();
    const startTime = now.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
    });
    const endTimeObj = new Date(now.getTime() + 60 * 60 * 1000);
    const endTime = endTimeObj.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
    });
    const dayOfWeek = now.toLocaleString("en-US", { weekday: "long" });

    setFormData((prev) => ({
      ...prev,
      date: now
        .toLocaleDateString("en-US", {
          month: "2-digit",
          day: "2-digit",
          year: "numeric",
        })
        .replace(/\//g, "/"),
      start_time: startTime,
      end_time: endTime,
      day_of_week: dayOfWeek,
    }));

    axios
      .get("http://localhost:8000/departments")
      .then((response) => setDepartments(response.data))
      .catch((error) => {
        console.error("Error fetching departments:", error);
        setError("Failed to load departments");
      });
  }, []);

  // Fetch years when dept_name changes
  useEffect(() => {
    if (formData.dept_name) {
      axios
        .get(`http://localhost:8000/years/${formData.dept_name}`)
        .then((response) => setYears(response.data))
        .catch((error) => {
          console.error("Error fetching years:", error);
          setError("Failed to load years");
        });
      setYears([]);
      setSections([]);
      setSubjects([]);
      setFormData((prev) => ({
        ...prev,
        year: "",
        section_name: "",
        subject_code: "",
      }));
    }
  }, [formData.dept_name]);

  // Fetch sections and subjects when year changes
  useEffect(() => {
    if (formData.dept_name && formData.year) {
      axios
        .get(
          `http://localhost:8000/sections/${formData.dept_name}/${formData.year}`
        )
        .then((response) => setSections(response.data))
        .catch((error) => {
          console.error("Error fetching sections:", error);
          setError("Failed to load sections");
        });
      axios
        .get(
          `http://localhost:8000/subjects/${formData.dept_name}/${formData.year}`
        )
        .then((response) => setSubjects(response.data))
        .catch((error) => {
          console.error("Error fetching subjects:", error);
          setError("Failed to load subjects");
        });
      setSections([]);
      setSubjects([]);
      setFormData((prev) => ({
        ...prev,
        section_name: "",
        subject_code: "",
      }));
    }
  }, [formData.dept_name, formData.year]);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError(null);
  };

  const handleNext = () => {
    if (!formData.dept_name || !formData.year || !formData.section_name) {
      setError("Please fill all class details");
      return;
    }
    setCurrentSlide(2);
    setError(null);
  };

  const handleBack = () => {
    setCurrentSlide(1);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (
      !formData.dept_name ||
      !formData.year ||
      !formData.section_name ||
      !formData.subject_code ||
      !formData.date ||
      !formData.start_time ||
      !formData.end_time ||
      !formData.day_of_week
    ) {
      setError("Please fill all required fields");
      return;
    }

    try {
      const response = await axios.post(
        "http://localhost:8000/create-timetable-slot",
        {
          dept_name: formData.dept_name,
          year: parseInt(formData.year),
          section_name: formData.section_name,
          subject_code: formData.subject_code,
          date: formData.date,
          start_time: formData.start_time,
          end_time: formData.end_time,
          day_of_week: formData.day_of_week,
        }
      );

      sessionStorage.setItem(
        "attendanceForm",
        JSON.stringify({
          ...formData,
          timetable_id: response.data.timetable_id,
        })
      );
      navigate("/attendance-assist");
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to create timetable slot");
    }
  };

  const handleViewReport = () => {
    navigate("/report");
  };

  return (
    <div className="min-h-screen bg-blue-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-lg">
        <div className="flex items-center justify-center mb-6">
          <h1 className="text-3xl font-bold text-gray-800">
            Attendance System
          </h1>
        </div>

        {/* Step Indicator */}
        <div className="flex justify-center mb-6">
          <div className="flex items-center gap-4">
            <div className="flex items-center">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-white ${
                  currentSlide === 1 ? "bg-blue-600" : "bg-gray-300"
                }`}
              >
                1
              </div>
              <span
                className={`ml-2 text-sm ${
                  currentSlide === 1 ? "text-gray-800 font-medium" : "text-gray-500"
                }`}
              >
                Class Details
              </span>
            </div>
            <div className="w-12 h-1 bg-gray-200 rounded-full"></div>
            <div className="flex items-center">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-white ${
                  currentSlide === 2 ? "bg-blue-600" : "bg-gray-300"
                }`}
              >
                2
              </div>
              <span
                className={`ml-2 text-sm ${
                  currentSlide === 2 ? "text-gray-800 font-medium" : "text-gray-500"
                }`}
              >
                Subject & Schedule
              </span>
            </div>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-3 bg-red-100 text-red-700 rounded-lg text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Slide 1: Class Details */}
          {currentSlide === 1 && (
            <div className="animate-slideIn">
              <h2 className="text-lg font-semibold text-gray-700 mb-4">
                Class Details
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-600">
                    Department
                  </label>
                  <select
                    name="dept_name"
                    value={formData.dept_name}
                    onChange={handleChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                    required
                  >
                    <option value="">Select Department</option>
                    {departments.map((dept) => (
                      <option key={dept.dept_name} value={dept.dept_name}>
                        {dept.dept_name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600">
                    Year
                  </label>
                  <select
                    name="year"
                    value={formData.year}
                    onChange={handleChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                    required
                  >
                    <option value="">Select Year</option>
                    {years.map((y) => (
                      <option key={y.year} value={y.year}>
                        {y.year}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600">
                    Section
                  </label>
                  <select
                    name="section_name"
                    value={formData.section_name}
                    onChange={handleChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                    required
                  >
                    <option value="">Select Section</option>
                    {sections.map((s) => (
                      <option key={s.section_id} value={s.section_name}>
                        {s.section_name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="mt-6 flex gap-4">
                <button
                  type="button"
                  onClick={handleNext}
                  className="flex-1 bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition flex items-center justify-center gap-2"
                >
                  Next <ArrowRight size={18} />
                </button>
                <button
                  type="button"
                  onClick={handleViewReport}
                  className="flex-1 bg-gray-600 text-white py-3 rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 transition flex items-center justify-center gap-2"
                >
                  <FileText size={18} />
                  View Report
                </button>
              </div>
            </div>
          )}

          {/* Slide 2: Subject & Schedule */}
          {currentSlide === 2 && (
            <div className="animate-slideIn">
              <h2 className="text-lg font-semibold text-gray-700 mb-4">
                Subject & Schedule
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-600">
                    Subject
                  </label>
                  <select
                    name="subject_code"
                    value={formData.subject_code}
                    onChange={handleChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                    required
                  >
                    <option value="">Select Subject</option>
                    {subjects.map((sub) => (
                      <option key={sub.subject_code} value={sub.subject_code}>
                        {sub.subject_name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600">
                    Date
                  </label>
                  <div className="mt-1 relative">
                    <input
                      type="text"
                      name="date"
                      value={formData.date}
                      className="block w-full border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-500 cursor-not-allowed"
                      readOnly
                    />
                    <Calendar
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                      size={20}
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600">
                    Start Time
                  </label>
                  <div className="mt-1 relative">
                    <input
                      type="text"
                      name="start_time"
                      value={formData.start_time}
                      className="block w-full border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-500 cursor-not-allowed"
                      readOnly
                    />
                    <Clock
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                      size={20}
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600">
                    End Time
                  </label>
                  <div className="mt-1 relative">
                    <input
                      type="text"
                      name="end_time"
                      value={formData.end_time}
                      className="block w-full border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-500 cursor-not-allowed"
                      readOnly
                    />
                    <Clock
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                      size={20}
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600">
                    Day of Week
                  </label>
                  <input
                    type="text"
                    name="day_of_week"
                    value={formData.day_of_week}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-500 cursor-not-allowed"
                    readOnly
                  />
                </div>
              </div>
              <div className="mt-6 flex gap-4">
                <button
                  type="button"
                  onClick={handleBack}
                  className="flex-1 bg-gray-200 text-gray-700 py-3 rounded-lg hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500 transition flex items-center justify-center gap-2"
                >
                  <ArrowLeft size={18} />
                  Back
                </button>
                <button
                  type="submit"
                  className="flex-1 bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                >
                  Take Attendance
                </button>
                <button
                  type="button"
                  onClick={handleViewReport}
                  className="flex-1 bg-gray-600 text-white py-3 rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 transition flex items-center justify-center gap-2"
                >
                  <FileText size={18} />
                  View Report
                </button>
              </div>
            </div>
          )}
        </form>
      </div>

      {/* Custom Animation for Slide */}
      <style>
        {`
          .animate-slideIn {
            animation: slideIn 0.3s ease-in-out;
          }
          @keyframes slideIn {
            from {
              opacity: 0;
              transform: translateX(20px);
            }
            to {
              opacity: 1;
              transform: translateX(0);
            }
          }
        `}
      </style>
    </div>
  );
};

export default AttendanceForm;