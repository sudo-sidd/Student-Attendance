import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Filter,
  Edit,
  Trash2,
  Plus,
  X,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import axios from "axios";

const AdminPage = () => {
  const navigate = useNavigate();
  const [filters, setFilters] = useState({
    dept_name: "",
    year: "",
    section_name: "",
    date: "",
  });
  const [attendanceData, setAttendanceData] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [years, setYears] = useState([]);
  const [sections, setSections] = useState([]);
  const [timetables, setTimetables] = useState([]);
  const [students, setStudents] = useState([]);
  const [error, setError] = useState(null);
  const [isFilterModalOpen, setIsFilterModalOpen] = useState(true);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  const [selectedAttendance, setSelectedAttendance] = useState(null);
  const [newAttendance, setNewAttendance] = useState({
    timetable_id: "",
    register_number: "",
    is_present: false,
  });

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
      setTimetables([]);
      setFilters((prev) => ({
        ...prev,
        year: "",
        section_name: "",
      }));
    }
  }, [filters.dept_name]);

  // Fetch sections and timetables when year changes
  useEffect(() => {
    if (filters.dept_name && filters.year) {
      axios
        .get(`http://localhost:8000/sections/${filters.dept_name}/${filters.year}`)
        .then((response) => setSections(response.data))
        .catch((err) => setError("Failed to fetch sections"));
      axios
        .get(
          `http://localhost:8000/timetables/${filters.dept_name}/${filters.year}`
        )
        .then((response) => setTimetables(response.data))
        .catch((err) => setError("Failed to fetch timetables"));
      setSections([]);
      setTimetables([]);
      setFilters((prev) => ({ ...prev, section_name: "" }));
    }
  }, [filters.dept_name, filters.year]);

  // Fetch students when section changes
  useEffect(() => {
    if (filters.dept_name && filters.year && filters.section_name) {
      axios
        .get(
          `http://localhost:8000/students/${filters.dept_name}/${filters.year}/${filters.section_name}`
        )
        .then((response) => setStudents(response.data))
        .catch((err) => setError("Failed to fetch students"));
    }
  }, [filters.dept_name, filters.year, filters.section_name]);

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
      setAttendanceData(response.data.attendance);
      setIsFilterModalOpen(false);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to fetch attendance");
    }
  };

  const resetFilters = () => {
    setFilters({
      dept_name: "",
      year: "",
      section_name: "",
      date: "",
    });
    setAttendanceData([]);
    setError(null);
  };

  const handleCreateChange = (e) => {
    const { name, value, type, checked } = e.target;
    setNewAttendance((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleCreateSubmit = async (e) => {
    e.preventDefault();
    if (!newAttendance.timetable_id || !newAttendance.register_number) {
      setError("Please select timetable and student");
      return;
    }

    try {
      await axios.post("http://localhost:8000/attendance", {
        timetable_id: parseInt(newAttendance.timetable_id),
        register_number: newAttendance.register_number,
        is_present: newAttendance.is_present,
      });
      setIsCreateModalOpen(false);
      setNewAttendance({ timetable_id: "", register_number: "", is_present: false });
      handleApplyFilters(); // Refresh data
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to create attendance");
    }
  };

  const handleEdit = (record) => {
    setSelectedAttendance(record);
    setNewAttendance({
      timetable_id: record.timetable_id,
      register_number: record.register_number,
      is_present: record.is_present,
    });
    setIsEditModalOpen(true);
  };

  const handleEditSubmit = async (e) => {
    e.preventDefault();
    try {
      await axios.put(
        `http://localhost:8000/attendance/${selectedAttendance.attendance_id}`,
        {
          timetable_id: parseInt(newAttendance.timetable_id),
          register_number: newAttendance.register_number,
          is_present: newAttendance.is_present,
        }
      );
      setIsEditModalOpen(false);
      setSelectedAttendance(null);
      handleApplyFilters(); // Refresh data
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to update attendance");
    }
  };

  const handleDelete = (record) => {
    setSelectedAttendance(record);
    setIsDeleteConfirmOpen(true);
  };

  const confirmDelete = async () => {
    try {
      await axios.delete(
        `http://localhost:8000/attendance/${selectedAttendance.attendance_id}`
      );
      setIsDeleteConfirmOpen(false);
      setSelectedAttendance(null);
      handleApplyFilters(); // Refresh data
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to delete attendance");
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm p-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-800">
          Admin - Manage Attendance
        </h1>
        <div className="flex gap-4">
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
            Form
          </button>
          <button
            onClick={() => navigate("/report")}
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
                d="M9 5l7 7-7 7"
              />
            </svg>
            Report
          </button>
        </div>
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
                  onClick={() => setIsFilterModalOpen(false)}
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

        {/* Create Modal */}
        {isCreateModalOpen && (
          <div className="fixed inset-0 bg-blue-50 bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl shadow-2xl p-8 w-full max-w-md animate-slideIn">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-gray-800">
                  Add Attendance
                </h2>
                <button
                  onClick={() => setIsCreateModalOpen(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <X size={24} />
                </button>
              </div>
              {error && <p className="text-red-500 text-sm mb-4">{error}</p>}
              <form onSubmit={handleCreateSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Timetable Slot
                  </label>
                  <select
                    name="timetable_id"
                    value={newAttendance.timetable_id}
                    onChange={handleCreateChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select Timetable</option>
                    {timetables.map((tt) => (
                      <option key={tt.timetable_id} value={tt.timetable_id}>
                        {tt.subject_name} ({tt.start_time}-{tt.end_time}, {tt.date})
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Student
                  </label>
                  <select
                    name="register_number"
                    value={newAttendance.register_number}
                    onChange={handleCreateChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select Student</option>
                    {students.map((student) => (
                      <option
                        key={student.register_number}
                        value={student.register_number}
                      >
                        {student.name} ({student.register_number})
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    name="is_present"
                    checked={newAttendance.is_present}
                    onChange={handleCreateChange}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <label className="ml-2 text-sm font-medium text-gray-700">
                    Present
                  </label>
                </div>
                <div className="flex gap-4">
                  <button
                    type="submit"
                    className="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
                  >
                    Add
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsCreateModalOpen(false)}
                    className="flex-1 bg-gray-200 text-gray-700 py-2 rounded-lg hover:bg-gray-300 transition"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Edit Modal */}
        {isEditModalOpen && (
          <div className="fixed inset-0 bg-blue-50 bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl shadow-2xl p-8 w-full max-w-md animate-slideIn">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-gray-800">
                  Edit Attendance
                </h2>
                <button
                  onClick={() => setIsEditModalOpen(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <X size={24} />
                </button>
              </div>
              {error && <p className="text-red-500 text-sm mb-4">{error}</p>}
              <form onSubmit={handleEditSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Timetable Slot
                  </label>
                  <select
                    name="timetable_id"
                    value={newAttendance.timetable_id}
                    onChange={handleCreateChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select Timetable</option>
                    {timetables.map((tt) => (
                      <option key={tt.timetable_id} value={tt.timetable_id}>
                        {tt.subject_name} ({tt.start_time}-{tt.end_time}, {tt.date})
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Student
                  </label>
                  <select
                    name="register_number"
                    value={newAttendance.register_number}
                    onChange={handleCreateChange}
                    className="mt-1 block w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select Student</option>
                    {students.map((student) => (
                      <option
                        key={student.register_number}
                        value={student.register_number}
                      >
                        {student.name} ({student.register_number})
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    name="is_present"
                    checked={newAttendance.is_present}
                    onChange={handleCreateChange}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <label className="ml-2 text-sm font-medium text-gray-700">
                    Present
                  </label>
                </div>
                <div className="flex gap-4">
                  <button
                    type="submit"
                    className="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
                  >
                    Save
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsEditModalOpen(false)}
                    className="flex-1 bg-gray-200 text-gray-700 py-2 rounded-lg hover:bg-gray-300 transition"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Delete Confirmation */}
        {isDeleteConfirmOpen && (
          <div className="fixed inset-0 bg-blue-50 bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl shadow-2xl p-8 w-full max-w-sm animate-slideIn">
              <div className="flex items-center gap-2 mb-4">
                <AlertCircle className="text-red-600" size={24} />
                <h2 className="text-lg font-semibold text-gray-800">
                  Confirm Delete
                </h2>
              </div>
              <p className="text-gray-600 mb-6">
                Are you sure you want to delete this attendance record for{" "}
                {selectedAttendance?.name}?
              </p>
              <div className="flex gap-4">
                <button
                  onClick={confirmDelete}
                  className="flex-1 bg-red-600 text-white py-2 rounded-lg hover:bg-red-700 transition"
                >
                  Delete
                </button>
                <button
                  onClick={() => setIsDeleteConfirmOpen(false)}
                  className="flex-1 bg-gray-200 text-gray-700 py-2 rounded-lg hover:bg-gray-300 transition"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-800">
              Attendance Records
            </h2>
            <div className="flex gap-4">
              <button
                onClick={() => setIsFilterModalOpen(true)}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center gap-2"
              >
                <Filter size={16} />
                Filters
              </button>
              <button
                onClick={() => setIsCreateModalOpen(true)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
              >
                <Plus size={16} />
                Add Attendance
              </button>
            </div>
          </div>

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
                    <th className="py-3 px-4 text-gray-700 font-semibold">
                      Actions
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
                      <td className="py-3 px-4">{record.subject_name}</td>
                      <td className="py-3 px-4">{record.date}</td>
                      <td className="py-3 px-4">
                        {record.start_time}-{record.end_time}
                      </td>
                      <td className="py-3 px-4">
                        {record.is_present ? (
                          <span className="text-green-600 font-medium">
                            Present
                          </span>
                        ) : (
                          <span className="text-red-600 font-medium">Absent</span>
                        )}
                      </td>
                      <td className="py-3 px-4 flex gap-2">
                        <button
                          onClick={() => handleEdit(record)}
                          className="p-1 text-blue-600 hover:text-blue-800"
                        >
                          <Edit size={18} />
                        </button>
                        <button
                          onClick={() => handleDelete(record)}
                          className="p-1 text-red-600 hover:text-red-800"
                        >
                          <Trash2 size={18} />
                        </button>
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

export default AdminPage;