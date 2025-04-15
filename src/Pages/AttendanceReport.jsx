import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Filter, Download, X, ChevronDown, ChevronUp } from "lucide-react";
import axios from "axios";
import * as XLSX from "xlsx";

export default function AttendanceReport() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState({
    dept_name: "",
    year: "",
    section_name: "",
    date: "",
  });
  const [attendanceData, setAttendanceData] = useState([]);
  const [expandedPeriods, setExpandedPeriods] = useState({});
  const [departments, setDepartments] = useState([]);
  const [years, setYears] = useState([]);
  const [sections, setSections] = useState([]);
  const [error, setError] = useState(null);
  const [isFilterModalOpen, setIsFilterModalOpen] = useState(true);
  const [formData, setFormData] = useState({ dept_name: "", date: "" });

  // Fetch departments on mount and load form data from sessionStorage
  useEffect(() => {
    // Load form data from sessionStorage
    const storedForm = sessionStorage.getItem("attendanceForm");
    if (storedForm) {
      const parsedForm = JSON.parse(storedForm);
      setFormData({
        dept_name: parsedForm.dept_name || "",
        date: parsedForm.date || "",
      });
    }

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
      setYears([]);
      setSections([]);
      setFilters((prev) => ({
        ...prev,
        year: "",
        section_name: "",
      }));
    }
  }, [filters.dept_name]);

  // Fetch sections when year changes
  useEffect(() => {
    if (filters.dept_name && filters.year) {
      axios
        .get(`http://localhost:8000/sections/${filters.dept_name}/${filters.year}`)
        .then((response) => setSections(response.data))
        .catch((err) => setError("Failed to fetch sections"));
      setSections([]);
      setFilters((prev) => ({ ...prev, section_name: "" }));
    }
  }, [filters.dept_name, filters.year]);

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters((prev) => ({ ...prev, [name]: value }));
    setError(null);
  };

  const handleApplyFilters = async () => {
    if (
      !filters.dept_name ||
      !filters.year ||
      !filters.section_name ||
      !filters.date
    ) {
      setError("Please fill all filter fields");
      return;
    }

    try {
      const params = {
        dept_name: filters.dept_name,
        year: parseInt(filters.year, 10),
        section_name: filters.section_name,
        date: filters.date,
      };

      const response = await axios.get("http://localhost:8000/get-attendance", {
        params,
      });

      // Group attendance by timetable_id
      const groupedData = response.data.attendance.reduce((acc, record) => {
        const key = record.timetable_id;
        if (!acc[key]) {
          acc[key] = {
            timetable_id: record.timetable_id,
            subject_code: record.subject_code,
            subject_name: record.subject_name,
            start_time: record.start_time,
            end_time: record.end_time,
            records: [],
          };
        }
        acc[key].records.push(record);
        return acc;
      }, {});

      const periods = Object.values(groupedData);
      setAttendanceData(periods);
      // Initialize all periods as expanded
      setExpandedPeriods(
        periods.reduce((acc, period) => {
          acc[period.timetable_id] = true;
          return acc;
        }, {})
      );
      // Update formData with applied filters
      setFormData({
        dept_name: filters.dept_name,
        date: filters.date,
      });
      setIsFilterModalOpen(false);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to fetch attendance");
    }
  };

  const togglePeriod = (timetable_id) => {
    setExpandedPeriods((prev) => ({
      ...prev,
      [timetable_id]: !prev[timetable_id],
    }));
  };

  const handleExport = () => {
    const data = [];
    attendanceData.forEach((period) => {
      // Add period header
      data.push({
        "Period": `Subject: ${period.subject_name} (${period.subject_code})`,
        "Time": `${period.start_time} - ${period.end_time}`,
      });
      // Add attendance records
      period.records.forEach((record) => {
        data.push({
          "Register Number": record.register_number,
          Name: record.name,
          Subject: record.subject_name,
          Date: record.date,
          Time: `${record.start_time} - ${record.end_time}`,
          Status: record.is_present ? "Present" : "Absent",
        });
      });
      // Add empty row for separation
      data.push({});
    });

    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Attendance Report");
    XLSX.writeFile(
      workbook,
      `Attendance_Report_${filters.date.replace(/\//g, "-")}.xlsx`
    );
  };

  const resetFilters = () => {
    setFilters({
      dept_name: "",
      year: "",
      section_name: "",
      date: "",
    });
    setAttendanceData([]);
    setExpandedPeriods({});
    setFormData({ dept_name: "", date: "" });
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm p-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-800">Attendance Report</h1>
        <button
          onClick={() => navigate("/")}
          className="px-4 py-2 text-gray-600 hover:text-gray-800 flex items-center gap-2 rounded-lg hover:bg-gray-100"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M15 19l-7-7 7-7"
            />
          </svg>
          Back to Home
        </button>
      </header>

      <main className="max-w-6xl mx-auto p-6">
        {/* Filter Modal */}
        {isFilterModalOpen && (
          <div className="fixed inset-0 bg-blue-50 bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl shadow-2xl p-8 w-full max-w-md animate-slideIn">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                  <Filter size={20} /> Select Filters
                </h2>
                <button
                  onClick={() => {
                    setIsFilterModalOpen(false);
                    navigate('/');
                  }}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <X size={24} />
                </button>
              </div>
              {error && <p className="text-red-500 text-sm mb-4">{error}</p>}
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Department
                  </label>
                  <select
                    name="dept_name"
                    value={filters.dept_name}
                    onChange={handleFilterChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
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
                  <label className="block text-sm font-medium text-gray-700">
                    Year
                  </label>
                  <select
                    name="year"
                    value={filters.year}
                    onChange={handleFilterChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
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
                  <label className="block text-sm font-medium text-gray-700">
                    Section
                  </label>
                  <select
                    name="section_name"
                    value={filters.section_name}
                    onChange={handleFilterChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select Section</option>
                    {sections.map((s) => (
                      <option key={s.section_id} value={s.section_name}>
                        {s.section_name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Date
                  </label>
                  <input
                    type="text"
                    name="date"
                    value={filters.date}
                    onChange={handleFilterChange}
                    placeholder="MM/DD/YYYY"
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
              <div className="mt-6 flex gap-4">
                <button
                  onClick={handleApplyFilters}
                  className="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
                >
                  Apply Filters
                </button>
                <button
                  onClick={resetFilters}
                  className="flex-1 bg-gray-200 text-gray-700 py-2 rounded-lg hover:bg-gray-300 transition"
                >
                  Reset
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-800">
              Attendance Records
            </h2>
            <div className="flex gap-4">
              <button
                onClick={() => setIsFilterModalOpen(true)}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center gap-2"
              >
                <Filter size={16} />
                Change Filters
              </button>
              {attendanceData.length > 0 && (
                <button
                  onClick={handleExport}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
                >
                  <Download size={16} />
                  Export to Excel
                </button>
              )}
            </div>
          </div>

          {/* Display Selected Department and Date */}
          {(formData.dept_name || formData.date) && (
            <div className="mb-6 p-4 bg-gray-50 rounded-lg animate-slideIn">
              <div className="flex gap-6">
                {formData.dept_name && (
                  <p className="text-gray-700">
                    <span className="font-medium">Department:</span>{" "}
                    {formData.dept_name}
                  </p>
                )}
                {formData.date && (
                  <p className="text-gray-700">
                    <span className="font-medium">Date:</span> {formData.date}
                  </p>
                )}
              </div>
            </div>
          )}

          {attendanceData.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-500 text-lg">
                Please apply filters to view attendance records
              </p>
              <button
                onClick={() => setIsFilterModalOpen(true)}
                className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Open Filters
              </button>
            </div>
          ) : (
            <div className="space-y-6">
              {attendanceData.map((period) => {
                const isExpanded = expandedPeriods[period.timetable_id];
                const presentCount = period.records.filter(
                  (r) => r.is_present
                ).length;
                const totalCount = period.records.length;
                return (
                  <div
                    key={period.timetable_id}
                    className="border border-gray-200 rounded-lg overflow-hidden"
                  >
                    <div className="bg-gray-50 p-4 flex justify-between items-center">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-800">
                          {period.subject_name} ({period.subject_code})
                        </h3>
                        <p className="text-gray-600">
                          Time: {period.start_time} - {period.end_time}
                        </p>
                      </div>
                      <button
                        onClick={() => togglePeriod(period.timetable_id)}
                        className="p-2 text-gray-600 hover:text-gray-800 rounded-full hover:bg-gray-200 transition"
                        aria-label={isExpanded ? "Collapse period" : "Expand period"}
                      >
                        {isExpanded ? (
                          <ChevronUp size={20} />
                        ) : (
                          <ChevronDown size={20} />
                        )}
                      </button>
                    </div>
                    {isExpanded && (
                      <div className="p-4 animate-slideIn">
                        <div className="mb-4 flex gap-4">
                          <span className="text-sm text-gray-600">
                            Present: <span className="text-green-600">{presentCount}</span>
                          </span>
                          <span className="text-sm text-gray-600">
                            Absent: <span className="text-red-600">{totalCount - presentCount}</span>
                          </span>
                          <span className="text-sm text-gray-600">
                            Total: {totalCount}
                          </span>
                        </div>
                        <div className="overflow-x-auto">
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
                                  Status
                                </th>
                              </tr>
                            </thead>
                            <tbody>
                              {period.records.map((record) => (
                                <tr
                                  key={record.attendance_id}
                                  className="border-t border-gray-200 hover:bg-gray-50"
                                >
                                  <td className="py-3 px-4">{record.register_number}</td>
                                  <td className="py-3 px-4">{record.name}</td>
                                  <td className="py-3 px-4">
                                    {record.is_present ? (
                                      <span className="text-green-600 font-medium">
                                        Present
                                      </span>
                                    ) : (
                                      <span className="text-red-600 font-medium">
                                        Absent
                                      </span>
                                    )}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
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
}