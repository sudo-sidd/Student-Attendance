import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Check, X, Trash2, Filter } from 'lucide-react';
import axios from 'axios';

export default function AdminPage() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState({
    dept_name: '',
    year: '',
    section_name: '',
    subject_code: '',
    date: '',
  });
  const [attendanceRecords, setAttendanceRecords] = useState([]);
  const [error, setError] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [editPresence, setEditPresence] = useState(null);

  const fetchAttendance = async () => {
    try {
      const params = {};
      if (filters.dept_name) params.dept_name = filters.dept_name;
      if (filters.year) params.year = parseInt(filters.year, 10);
      if (filters.section_name) params.section_name = filters.section_name;
      if (filters.subject_code) params.subject_code = filters.subject_code;
      if (filters.date) params.date = filters.date;

      const response = await axios.get('http://localhost:8000/get-attendance', { params });
      setAttendanceRecords(response.data.attendance);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch attendance');
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

  const handleDelete = async (attendance_id) => {
    if (!window.confirm('Are you sure you want to delete this record?')) return;
    try {
      await axios.delete(`http://localhost:8000/delete-attendance/${attendance_id}`);
      setAttendanceRecords((prev) => prev.filter((record) => record.attendance_id !== attendance_id));
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to delete record');
    }
  };

  const handleEdit = (record) => {
    setEditingId(record.attendance_id);
    setEditPresence(record.is_present);
  };

  const handleSaveEdit = async (attendance_id, register_number, section_name, subject_code, date, start_time, end_time) => {
    try {
      // Re-submit attendance with updated is_present
      const formDataToSend = new FormData();
      formDataToSend.append('dept_name', filters.dept_name || 'AIML');
      formDataToSend.append('year', filters.year || '2');
      formDataToSend.append('section_name', section_name);
      formDataToSend.append('subject_code', subject_code);
      formDataToSend.append('date', date);
      formDataToSend.append('start_time', start_time);
      formDataToSend.append('end_time', end_time);
      formDataToSend.append('attendance', JSON.stringify([{
        register_number,
        name: attendanceRecords.find(r => r.attendance_id === attendance_id).name,
        is_present: editPresence
      }]));

      await axios.post('http://localhost:8000/submit-attendance', formDataToSend);
      setAttendanceRecords((prev) =>
        prev.map((record) =>
          record.attendance_id === attendance_id ? { ...record, is_present: editPresence } : record
        )
      );
      setEditingId(null);
      setEditPresence(null);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to update record');
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditPresence(null);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="border-b border-gray-200 p-4 flex justify-between items-center bg-white shadow-sm">
        <h1 className="text-xl font-semibold text-gray-800">Admin Dashboard</h1>
        <button
          onClick={() => navigate('/attendance-assist')}
          className="p-2 text-gray-600 hover:text-gray-800 flex items-center gap-2"
        >
          <ArrowLeft size={24} />
          Back to Upload
        </button>
      </header>

      <main className="p-6 max-w-4xl mx-auto">
        {error && <p className="text-red-500 text-center mb-4">{error}</p>}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Filter size={20} /> Filter Attendance
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Department</label>
              <input
                type="text"
                name="dept_name"
                value={filters.dept_name}
                onChange={handleFilterChange}
                className="mt-1 block w-full border border-gray-300 rounded-md p-2"
                placeholder="e.g., AIML"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Year</label>
              <input
                type="number"
                name="year"
                value={filters.year}
                onChange={handleFilterChange}
                className="mt-1 block w-full border border-gray-300 rounded-md p-2"
                placeholder="e.g., 2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Section</label>
              <input
                type="text"
                name="section_name"
                value={filters.section_name}
                onChange={handleFilterChange}
                className="mt-1 block w-full border border-gray-300 rounded-md p-2"
                placeholder="e.g., B"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Subject Code</label>
              <input
                type="text"
                name="subject_code"
                value={filters.subject_code}
                onChange={handleFilterChange}
                className="mt-1 block w-full border border-gray-300 rounded-md p-2"
                placeholder="e.g., AI201"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Date</label>
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

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Attendance Records</h2>
          {attendanceRecords.length === 0 ? (
            <p className="text-gray-600 text-center">No attendance records found</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left rounded-lg">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="py-3 px-4 text-gray-700 font-semibold rounded-tl-lg">Register Number</th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">Name</th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">Section</th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">Subject</th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">Date</th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">Time</th>
                    <th className="py-3 px-4 text-gray-700 font-semibold">Status</th>
                    <th className="py-3 px-4 text-gray-700 font-semibold rounded-tr-lg">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {attendanceRecords.map((record) => (
                    <tr key={record.attendance_id} className="border-b last:border-0 hover:bg-gray-50">
                      <td className="py-3 px-4">{record.register_number}</td>
                      <td className="py-3 px-4">{record.name}</td>
                      <td className="py-3 px-4">{record.section_name}</td>
                      <td className="py-3 px-4">{record.subject_code}</td>
                      <td className="py-3 px-4">{record.date}</td>
                      <td className="py-3 px-4">{record.start_time} - {record.end_time}</td>
                      <td className="py-3 px-4">
                        {editingId === record.attendance_id ? (
                          <select
                            value={editPresence}
                            onChange={(e) => setEditPresence(parseInt(e.target.value, 10))}
                            className="border border-gray-300 rounded-md p-1"
                          >
                            <option value={1}>Present</option>
                            <option value={0}>Absent</option>
                          </select>
                        ) : record.is_present ? (
                          <span className="text-green-600 flex items-center gap-1">
                            <Check size={16} /> Present
                          </span>
                        ) : (
                          <span className="text-red-600 flex items-center gap-1">
                            <X size={16} /> Absent
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 flex gap-2">
                        {editingId === record.attendance_id ? (
                          <>
                            <button
                              onClick={() => handleSaveEdit(
                                record.attendance_id,
                                record.register_number,
                                record.section_name,
                                record.subject_code,
                                record.date,
                                record.start_time,
                                record.end_time
                              )}
                              className="text-blue-600 hover:text-blue-800"
                            >
                              Save
                            </button>
                            <button
                              onClick={handleCancelEdit}
                              className="text-gray-600 hover:text-gray-800"
                            >
                              Cancel
                            </button>
                          </>
                        ) : (
                          <>
                            <button
                              onClick={() => handleEdit(record)}
                              className="text-yellow-600 hover:text-yellow-800"
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => handleDelete(record.attendance_id)}
                              className="text-red-600 hover:text-red-800"
                            >
                              <Trash2 size={16} />
                            </button>
                          </>
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