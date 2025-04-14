import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Filter, Download } from "lucide-react";
import axios from "axios";
import * as XLSX from "xlsx";

export default function AttendanceReport() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState({
    dept_name: "",
    year: "",
    section_name: "",
    subject_code: "",
    date: "",
  });
  const [attendanceRecords, setAttendanceRecords] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [years, setYears] = useState([]);
  const [sections, setSections] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [error, setError] = useState(null);

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
      setYears([]);
      setSections([]);
      setSubjects([]);
      setFilters((prev) => ({
        ...prev,
        year: "",
        section_name: "",
        subject_code: "",
      }));
    }
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
      setSections([]);
      setSubjects([]);
      setFilters((prev) => ({
        ...prev,
        section_name: "",
        subject_code: "",
      }));
    }
  }, [filters.dept_name, filters.year]);

  const fetchAttendance = async () => {
    try {
      const params = {};
      if (filters.dept_name) params.dept_name = filters.dept_name;
      if (filters.year) params.year = parseInt(filters.year, 10);
      if (filters.section_name) params.section_name = filters.section_name;
      if (filters.subject_code) params.subject_code = filters.subject_code;
      if (filters.date) params.date = filters.date;

      const response = await axios.get("http://localhost:8000/get-attendance", {
        params,
      });
      setAttendanceRecords(response.data.attendance);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to fetch attendance");
    }
  };

  useEffect(() => {
    fetchAttendance();
  }, []);

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters((prev) => ({ ...prev, [name]: value }));
  };

  const handleApplyFilters = () => {
    fetchAttendance();
  };

  const handleExport = () => {
    const data = attendanceRecords.map((record) => ({
      "Register Number": record.register_number,
      Name: record.name,
      Section: record.section_name,
      Subject: record.subject_name,
      Date: record.date,
      Time: `${record.start_time} - ${record.end_time}`,
      Status: record.is_present ? "Present" : "Absent",
    }));

    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Attendance Report");
    XLSX.writeFile(workbook, `Attendance_Report_${new Date().toISOString()}.xlsx`);
  };

  const totalStudents = attendanceRecords.length;
  const presentCount = attendanceRecords.filter(
    (r) => r.is_present
  ).length;
  const absentCount = totalStudents - presentCount;

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="border-b border-gray-200 p-4 flex justify-between items-center bg-white shadow-sm">
        <h1 className="text-xl font-semibold text-gray-800">
          Attendance Report
        </h1>
        <button
          onClick={() => navigate("/attendance-assist")}
          className="p-2 text-gray-600 hover:text-gray-800 flex items-center gap-2"
        >
          Back to Upload
        </button>
      </header>

      <main className="p-6 max-w-4xl mx-auto">
        {error && <p className="text-red-500 text-center mb-4">{error}</p>}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Filter size={20} /> Filter Report
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Department
              </label>
              <select
                name="dept_name"
                value={filters.dept_name}
                onChange={handleFilterChange}
                className="mt-1 block w-full border border-gray-300 rounded-md p-2"
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
                className="mt-1 block w-full border border-gray-300 rounded-md p-2"
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
                className="mt-1 block w-full border border-gray-300 rounded-md p-2"
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
                Subject
              </label>
              <select
                name="subject_code"
                value={filters.subject_code}
                onChange={handleFilterChange}
                className="mt-1 block w-full border border-gray-300 rounded-md p-2"
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
              <label className="block text-sm font-medium text-gray-700">
                Date
              </label>
              <input
                type="text"
                name="date"
                value={filters.date}
                onChange={handleFilterChange}
                className="mt-1 block w-full border border-gray-300 rounded-md p-2"
                placeholder="MM/DD/YYYY"
              />
            </div>
          </div>
          <button
            onClick={handleApplyFilters}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          >
            Apply Filters
          </button>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            Attendance Summary
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-gray-600">Total Students</p>
              <p className="text-2xl font-semibold text-gray-800">
                {totalStudents}
              </p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <p className="text-gray-600">Present</p>
              <p className="text-2xl font-semibold text-green-600">
                {presentCount}
              </p>
            </div>
            <div className="text-center p-4 bg-red-50 rounded-lg">
              <p className="text-gray-600">Absent</p>
              <p className="text-2xl font-semibold text-red-600">
                {absentCount}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold text-gray-800">
              Detailed Attendance
            </h2>
            <button
              onClick={handleExport}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
            >
              <Download size={16} />
              Export to Excel
            </button>
          </div>
          {attendanceRecords.length === 0 ? (
            <p className="text-gray-600 text-center">
              No attendance records found
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left rounded-lg">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="py-3 px-4 text-gray-700 font-semibold rounded-tl-lg">
                      Register Number
                    </th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">
                      Name
                    </th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">
                      Section
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
                    <th className="py-3 px-4 text-gray-700 font-semibold rounded-tr-lg">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {attendanceRecords.map((record) => (
                    <tr
                      key={record.attendance_id}
                      className="border-b last:border-0 hover:bg-gray-50"
                    >
                      <td className="py-3 px-4">{record.register_number}</td>
                      <td className="py-3 px-4">{record.name}</td>
                      <td className="py-3 px-4">{record.section_name}</td>
                      <td className="py-3 px-4">{record.subject_name}</td>
                      <td className="py-3 px-4">{record.date}</td>
                      <td className="py-3 px-4">
                        {record.start_time} - {record.end_time}
                      </td>
                      <td className="py-3 px-4">
                        {record.is_present ? (
                          <span className="text-green-600">Present</span>
                        ) : (
                          <span className="text-red-600">Absent</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}