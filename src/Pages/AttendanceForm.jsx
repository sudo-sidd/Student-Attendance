import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
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
    time: "",
  });
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // Set current date and time on mount
  useEffect(() => {
    const now = new Date();
    setFormData((prev) => ({
      ...prev,
      date: now
        .toLocaleDateString("en-US", {
          month: "2-digit",
          day: "2-digit",
          year: "numeric",
        })
        .replace(/\//g, "/"),
      time: now.toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
    }));

    // Fetch departments
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
      // Reset dependent fields
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
      // Reset dependent fields
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
    setError(null); // Clear error on change
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (
      !formData.dept_name ||
      !formData.year ||
      !formData.section_name ||
      !formData.subject_code ||
      !formData.date ||
      !formData.time
    ) {
      setError("Please fill all fields");
      return;
    }

    sessionStorage.setItem("attendanceForm", JSON.stringify(formData));
    navigate("/attendance-assist");
  };

  return (
    <div className="bg-blue-50 min-h-screen flex justify-center items-center">
      <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full">
        <h1 className="text-3xl font-bold text-center text-blue-600 mb-6">
          Student Attendance System
        </h1>
        {error && <p className="text-red-500 text-center mb-4">{error}</p>}
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-lg font-medium text-gray-700">
              Department
            </label>
            <select
              name="dept_name"
              value={formData.dept_name}
              onChange={handleChange}
              className="w-full p-2 mt-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
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

          <div className="mb-4">
            <label className="block text-lg font-medium text-gray-700">
              Year
            </label>
            <select
              name="year"
              value={formData.year}
              onChange={handleChange}
              className="w-full p-2 mt-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
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

          <div className="mb-4">
            <label className="block text-lg font-medium text-gray-700">
              Section
            </label>
            <select
              name="section_name"
              value={formData.section_name}
              onChange={handleChange}
              className="w-full p-2 mt-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
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

          <div className="mb-4">
            <label className="block text-lg font-medium text-gray-700">
              Subject
            </label>
            <select
              name="subject_code"
              value={formData.subject_code}
              onChange={handleChange}
              className="w-full p-2 mt-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
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

          <div className="mb-4">
            <label className="block text-lg font-medium text-gray-700">
              Date
            </label>
            <input
              type="text"
              name="date"
              value={formData.date}
              onChange={handleChange}
              className="w-full p-2 mt-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              readOnly
            />
          </div>

          <div className="mb-6">
            <label className="block text-lg font-medium text-gray-700">
              Time
            </label>
            <input
              type="text"
              name="time"
              value={formData.time}
              onChange={handleChange}
              className="w-full p-2 mt-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              readOnly
            />
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Take Attendance
          </button>
        </form>
      </div>
    </div>
  );
};

export default AttendanceForm;