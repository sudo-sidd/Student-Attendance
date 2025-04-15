import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Calendar, Clock, FileText, ArrowRight, ArrowLeft } from "lucide-react";
import { NavLink } from 'react-router-dom';
import { Home, CheckCircle, Shield } from 'lucide-react';
import axios from "axios";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import Header from "../components/Header";
import Footer from "../components/Footer";

const AttendanceForm = () => {
  const [departments, setDepartments] = useState([]);
  const [years, setYears] = useState([]);
  const [sections, setSections] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [timeBlocks, setTimeBlocks] = useState([]);
  const [selectedTimeBlock, setSelectedTimeBlock] = useState(null);
  const [formData, setFormData] = useState({
    dept_name: "",
    year: "",
    section_name: "",
    subject_code: "",
    date: new Date(),
    start_time: "",
    end_time: "",
    time_block_id: "",
  });
  const [error, setError] = useState(null);
  const [currentSlide, setCurrentSlide] = useState(1); // Track slide (1 or 2)
  const navigate = useNavigate();

  // Helper to convert time string (HH:MM) to minutes
  const convertTimeStringToMinutes = (timeString) => {
    let [hours, minutes] = timeString.split(":");
    if (timeString.toLowerCase().includes("pm") && parseInt(hours) < 12) {
      hours = parseInt(hours) + 12;
    } else if (timeString.toLowerCase().includes("am") && parseInt(hours) === 12) {
      hours = 0;
    }
    hours = parseInt(hours);
    minutes = parseInt(minutes) || 0;
    return hours * 60 + minutes;
  };

  // Set current date and time on mount
  useEffect(() => {
    const now = new Date();

    setFormData((prev) => ({
      ...prev,
      date: now,
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

  const handleDateChange = (date) => {
    setFormData((prev) => ({
      ...prev,
      date: date,
    }));
    setError(null);
  };

  const handleTimeBlockChange = (e) => {
    const blockId = parseInt(e.target.value);
    const block = timeBlocks.find((b) => b.time_block_id === blockId);

    if (block) {
      setSelectedTimeBlock(block);
      setFormData((prev) => ({
        ...prev,
        start_time: block.start_time,
        end_time: block.end_time,
        time_block_id: block.time_block_id,
      }));
    }
  };

  const handleNext = () => {
    if (!formData.dept_name || !formData.year || !formData.section_name) {
      setError("Please fill all class details");
      return;
    }

    const batchYear = parseInt(formData.year);

    axios
      .get(`http://localhost:8000/time-blocks/${batchYear}`)
      .then((response) => {
        setTimeBlocks(response.data);

        const now = new Date();
        const currentTime = now.toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          hour12: false,
        });

        const sortedTimeBlocks = [...response.data].sort((a, b) => {
          return (
            convertTimeStringToMinutes(a.start_time) -
            convertTimeStringToMinutes(b.start_time)
          );
        });

        const currentBlock = sortedTimeBlocks.find((block) => {
          const startMinutes = convertTimeStringToMinutes(block.start_time);
          const endMinutes = convertTimeStringToMinutes(block.end_time);
          const currentMinutes = convertTimeStringToMinutes(currentTime);

          return currentMinutes >= startMinutes && currentMinutes < endMinutes;
        });

        if (currentBlock) {
          setSelectedTimeBlock(currentBlock);
          setFormData((prev) => ({
            ...prev,
            start_time: currentBlock.start_time,
            end_time: currentBlock.end_time,
            time_block_id: currentBlock.time_block_id,
          }));
        } else {
          const upcomingBlock = sortedTimeBlocks.find(
            (block) =>
              convertTimeStringToMinutes(block.start_time) >
              convertTimeStringToMinutes(currentTime)
          );

          const blockToUse =
            upcomingBlock || (sortedTimeBlocks.length > 0 ? sortedTimeBlocks[0] : null);

          if (blockToUse) {
            setSelectedTimeBlock(blockToUse);
            setFormData((prev) => ({
              ...prev,
              start_time: blockToUse.start_time,
              end_time: blockToUse.end_time,
              time_block_id: blockToUse.time_block_id,
            }));
          }
        }

        setCurrentSlide(2);
        setError(null);
      })
      .catch((error) => {
        console.error("Error fetching time blocks:", error);
        setError("Failed to load time blocks for selected year");
      });
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
      !formData.time_block_id
    ) {
      setError("Please fill all required fields");
      return;
    }

    try {
      const formattedDate = formData.date
        .toLocaleDateString("en-US", {
          month: "2-digit",
          day: "2-digit",
          year: "numeric",
        })
        .replace(/\//g, "/");

      const response = await axios.post(
        "http://localhost:8000/create-timetable-slot",
        {
          dept_name: formData.dept_name,
          year: parseInt(formData.year),
          section_name: formData.section_name,
          subject_code: formData.subject_code,
          date: formattedDate,
          start_time: formData.start_time,
          end_time: formData.end_time,
          time_block_id: formData.time_block_id,
        }
      );

      sessionStorage.setItem(
        "attendanceForm",
        JSON.stringify({
          ...formData,
          date: formattedDate,
          timetable_id: response.data.timetable_id,
        })
      );
      navigate("/attendance-assist");
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to create timetable slot");
    }
  };

  const handleViewReport = () => {
    navigate("/admin");
  };

  return (
    <>
      <Header />
      <div className="mt-3 flex-grow pt-24 pb-24 min-h-screen bg-blue-50 flex items-center justify-center p-4">
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
                    currentSlide === 1
                      ? "text-gray-800 font-medium"
                      : "text-gray-500"
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
                    currentSlide === 2
                      ? "text-gray-800 font-medium"
                      : "text-gray-500"
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

                  {/* Current Period Information */}
                  {selectedTimeBlock && (
                    <div className="mb-4 p-4 bg-blue-50 border border-blue-100 rounded-lg">
                      <h3 className="text-lg font-semibold text-blue-800 mb-1">
                        Current Period
                      </h3>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-gray-800">
                            Period {selectedTimeBlock.block_number}
                          </p>
                          <p className="text-gray-800">
                            {selectedTimeBlock.start_time} - {selectedTimeBlock.end_time}
                          </p>
                        </div>
                        <div className="text-sm text-gray-500">
                          Automatically selected based on current time
                        </div>
                      </div>
                    </div>
                  )}

                  <div>
                    <label className="block text-sm font-medium text-gray-600">
                      Date
                    </label>
                    <div className="mt-1 relative">
                      <DatePicker
                        selected={formData.date}
                        onChange={handleDateChange}
                        className="block w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                        dateFormat="MM/dd/yyyy"
                        placeholderText="Select Date"
                        required
                      />
                      <Calendar
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                        size={20}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-600">
                        Start Time
                      </label>
                      <div className="mt-1 relative">
                        <input
                          type="text"
                          name="start_time"
                          value={formData.start_time}
                          className="block w-full border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-700"
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
                          className="block w-full border border-gray-300 rounded-lg p-3 bg-gray-50 text-gray-700"
                          readOnly
                        />
                        <Clock
                          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                          size={20}
                        />
                      </div>
                    </div>
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
      <Footer />
    </>
  );
};

export default AttendanceForm;