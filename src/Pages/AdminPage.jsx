import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";

export default function AdminPage() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState({
    dept_name: "",
    year: "",
    section_name: "",
    subject_code: "",
    date: null, // Changed to null for DatePicker
  });
  const [attendanceData, setAttendanceData] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [years, setYears] = useState([]);
  const [sections, setSections] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [initialLoad, setInitialLoad] = useState(true);

  // Fetch departments on mount
  useEffect(() => {
    axios
      .get("http://localhost:8000/departments")
      .then((response) => setDepartments(response.data))
      .catch((err) => setError("Failed to fetch departments"));
  }, []);

  // Fetch years when dept_name changes
  useEffect(() => {
    if (filters.dept_name) {
      axios
        .get(`http://localhost:8000/years/${filters.dept_name}`)
        .then((response) => setYears(response.data))
        .catch((err) => setError("Failed to fetch years"));
    } else {
      setYears([]);
    }

    setSections([]);
    setSubjects([]);
    setFilters((prev) => ({
      ...prev,
      year: "",
      section_name: "",
      subject_code: "",
    }));
  }, [filters.dept_name]);

  // Fetch sections and subjects when year changes
  useEffect(() => {
    if (filters.dept_name && filters.year) {
      axios
        .get(`http://localhost:8000/sections/${filters.dept_name}/${filters.year}`)
        .then((response) => setSections(response.data))
        .catch((err) => setError("Failed to fetch sections"));

      axios
        .get(`http://localhost:8000/subjects/${filters.dept_name}/${filters.year}`)
        .then((response) => setSubjects(response.data))
        .catch((err) => setError("Failed to fetch subjects"));
    } else {
      setSections([]);
      setSubjects([]);
    }

    setFilters((prev) => ({
      ...prev,
      section_name: "",
      subject_code: "",
    }));
  }, [filters.dept_name, filters.year]);

  // Load all attendance data on initial load
  useEffect(() => {
    if (initialLoad) {
      fetchAllAttendanceData();
    }
  }, [initialLoad]);

  const fetchAllAttendanceData = async () => {
    setLoading(true);
    try {
      const response = await axios.get("http://localhost:8000/get-attendance");

      // Sort by date and time (newest first)
      const sortedData = response.data.attendance.sort((a, b) => {
        const dateCompare = new Date(b.date) - new Date(a.date);
        if (dateCompare !== 0) return dateCompare;
        return b.start_time.localeCompare(a.start_time);
      });

      setAttendanceData(sortedData);
      setInitialLoad(false);
      setLoading(false);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to fetch attendance data");
      setInitialLoad(false);
      setLoading(false);
    }
  };

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters((prev) => ({ ...prev, [name]: value }));
    setError(null);
  };

  const handleDateChange = (date) => {
    setFilters(prev => ({
      ...prev,
      date: date
    }));
  };

  const handleApplyFilters = async () => {
    setLoading(true);
    setError(null);

    const params = {};
    if (filters.dept_name) params.dept_name = filters.dept_name;
    if (filters.year) params.year = parseInt(filters.year, 10);
    if (filters.section_name) params.section_name = filters.section_name;
    if (filters.subject_code) params.subject_code = filters.subject_code;
    if (filters.date) {
      // Format the date for the API
      params.date = filters.date.toLocaleDateString("en-US", {
        month: "2-digit",
        day: "2-digit",
        year: "numeric",
      });
    }

    try {
      const response = await axios.get("http://localhost:8000/get-attendance", { params });

      // Sort by date and time (newest first)
      const sortedData = response.data.attendance.sort((a, b) => {
        const dateCompare = new Date(b.date) - new Date(a.date);
        if (dateCompare !== 0) return dateCompare;
        return b.start_time.localeCompare(a.start_time);
      });

      setAttendanceData(sortedData);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to fetch attendance");
    } finally {
      setLoading(false);
    }
  };

  const resetFilters = () => {
    setFilters({
      dept_name: "",
      year: "",
      section_name: "",
      date: null,
      subject_code: "",
    });
    fetchAllAttendanceData();
  };

  const exportAttendanceData = (format = 'csv') => {
    if (attendanceData.length === 0) {
      setError("No data to export");
      return;
    }

    // Format the data for export
    const dataToExport = attendanceData.map(record => ({
      'Register Number': record.register_number,
      'Name': record.name,
      'Subject': `${record.subject_name} (${record.subject_code})`,
      'Date': record.date,
      'Time': `${record.start_time}-${record.end_time}`,
      'Status': record.is_present ? 'Present' : 'Absent'
    }));

    // Create workbook and worksheet
    const workbook = XLSX.utils.book_new();
    const worksheet = XLSX.utils.json_to_sheet(dataToExport);

    // Set column widths
    const colWidths = [
      { wch: 15 }, // Register Number
      { wch: 20 }, // Name
      { wch: 30 }, // Subject
      { wch: 12 }, // Date
      { wch: 15 }, // Time
      { wch: 10 }  // Status
    ];
    worksheet['!cols'] = colWidths;

    // Generate file name based on filters
    let fileName = 'attendance';
    if (filters.dept_name) fileName += `_${filters.dept_name}`;
    if (filters.year) fileName += `_Year${filters.year}`;
    if (filters.section_name) fileName += `_${filters.section_name}`;
    if (filters.subject_code) fileName += `_${filters.subject_code}`;
    fileName += `.${format}`;

    // Export the file
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Attendance');
    const fileData = XLSX.write(workbook, { bookType: format, type: 'array' });
    const blob = new Blob([fileData], { type: 'application/octet-stream' });
    saveAs(blob, fileName);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm p-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-800">
          Admin - Attendance Records
        </h1>
        <div className="flex gap-4">
          <button
            onClick={() => navigate("/")}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
          >
            Home
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto p-6">
        {/* Filter Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Filter Attendance Records
          </h2>
          {error && <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">{error}</div>}

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Department
              </label>
              <select
                name="dept_name"
                value={filters.dept_name}
                onChange={handleFilterChange}
                className="w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Departments</option>
                {departments.map((dept) => (
                  <option key={dept.dept_name} value={dept.dept_name}>
                    {dept.dept_name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Year
              </label>
              <select
                name="year"
                value={filters.year}
                onChange={handleFilterChange}
                className="w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                disabled={!filters.dept_name}
              >
                <option value="">All Years</option>
                {years.map((y) => (
                  <option key={y.year} value={y.year}>
                    {y.year}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Section
              </label>
              <select
                name="section_name"
                value={filters.section_name}
                onChange={handleFilterChange}
                className="w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                disabled={!filters.year}
              >
                <option value="">All Sections</option>
                {sections.map((s) => (
                  <option key={s.section_id} value={s.section_name}>
                    {s.section_name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Subject
              </label>
              <select
                name="subject_code"
                value={filters.subject_code}
                onChange={handleFilterChange}
                className="w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                disabled={!filters.year}
              >
                <option value="">All Subjects</option>
                {subjects.map((subject) => (
                  <option key={subject.subject_code} value={subject.subject_code}>
                    {subject.subject_name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Date
              </label>
              <DatePicker
                selected={filters.date}
                onChange={handleDateChange}
                className="w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                placeholderText="All Dates"
                dateFormat="MM/dd/yyyy"
                isClearable
              />
            </div>
          </div>

          <div className="flex gap-4">
            <button
              onClick={handleApplyFilters}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center gap-2"
              disabled={loading}
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Loading...
                </>
              ) : (
                "Apply Filters"
              )}
            </button>
            <button
              onClick={resetFilters}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
            >
              Reset
            </button>
            <button
              onClick={() => exportAttendanceData('csv')}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
            >
              Export CSV
            </button>
            <button
              onClick={() => exportAttendanceData('xlsx')}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
            >
              Export Excel
            </button>
          </div>
        </div>

        {/* Active Filters Display */}
        {(filters.dept_name || filters.year || filters.section_name || filters.date || filters.subject_code) && (
          <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-100">
            <h3 className="font-medium text-blue-800 mb-2">Active Filters:</h3>
            <div className="flex flex-wrap gap-2">
              {filters.dept_name && (
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                  Department: {filters.dept_name}
                </span>
              )}
              {filters.year && (
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                  Year: {filters.year}
                </span>
              )}
              {filters.section_name && (
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                  Section: {filters.section_name}
                </span>
              )}
              {filters.subject_code && (
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                  Subject: {subjects.find(s => s.subject_code === filters.subject_code)?.subject_name || filters.subject_code}
                </span>
              )}
              {filters.date && (
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                  Date: {filters.date.toLocaleDateString("en-US", {
                    month: "2-digit",
                    day: "2-digit",
                    year: "numeric",
                  })}
                </span>
              )}
            </div>
          </div>
        )}

        {/* Attendance Records Table */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-800">
              Attendance Records
            </h2>
            <div className="text-gray-600">
              {attendanceData.length > 0 && `${attendanceData.length} records found`}
            </div>
          </div>

          {loading ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
              <p className="text-gray-500">Loading attendance records...</p>
            </div>
          ) : attendanceData.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-500 text-lg">
                No attendance records found for your selection
              </p>
              <p className="text-gray-400 mt-2">
                Try adjusting your filters or reset to view all records
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto animate-slideIn">
              <table className="w-full text-left">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="py-3 px-4 text-gray-700 font-semibold">
                      Register Number
                    </th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">
                      Name
                    </th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">
                      Subject
                    </th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">
                      Date
                    </th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">
                      Time
                    </th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {attendanceData.map((record) => (
                    <tr
                      key={record.attendance_id}
                      className="border-t border-gray-200 hover:bg-gray-50"
                    >
                      <td className="py-3 px-4">{record.register_number}</td>
                      <td className="py-3 px-4">{record.name}</td>
                      <td className="py-3 px-4">{record.subject_name} ({record.subject_code})</td>
                      <td className="py-3 px-4">{record.date}</td>
                      <td className="py-3 px-4">
                        {record.start_time}-{record.end_time}
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`px-3 py-1 rounded-full text-sm font-medium ${
                            record.is_present
                              ? "bg-green-100 text-green-800"
                              : "bg-red-100 text-red-800"
                          }`}
                        >
                          {record.is_present ? "Present" : "Absent"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>

      {/* Custom Animation */}
      <style>
        {`
          .animate-slideIn {
            animation: slideIn 0.3s ease-in-out;
          }
          @keyframes slideIn {
            from {
              opacity: 0;
              transform: translateY(10px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}
      </style>
    </div>
  );
}