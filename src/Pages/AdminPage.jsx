import React, { useState, useEffect } from "react";
import axios from "axios";
import { Filter, ChevronDown, ChevronUp, ChevronRight, Calendar } from "lucide-react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import Header from "../components/Header";
import Footer from "../components/Footer";

export default function AdminPage() {
  const [filters, setFilters] = useState({
    dept_name: "",
    year: "",
    section_name: "",
    subject_code: "",
    date: null,
  });
  
  const [attendanceData, setAttendanceData] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [years, setYears] = useState([]);
  const [sections, setSections] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isFilterVisible, setIsFilterVisible] = useState(true);
  const [groupedData, setGroupedData] = useState({});
  const [expandedGroups, setExpandedGroups] = useState({});
  
  // Fetch departments on component mount
  useEffect(() => {
    axios.get("http://localhost:8000/departments")
      .then(response => setDepartments(response.data))
      .catch(error => console.error("Error fetching departments:", error));
  }, []);

  // Fetch years when department changes
  useEffect(() => {
    if (filters.dept_name) {
      axios.get(`http://localhost:8000/years/${filters.dept_name}`)
        .then(response => setYears(response.data))
        .catch(error => console.error("Error fetching years:", error));
      
      // Reset year and section when department changes
      setFilters(prev => ({
        ...prev,
        year: "",
        section_name: "",
        subject_code: ""
      }));
      setSections([]);
      setSubjects([]);
    }
  }, [filters.dept_name]);

  // Fetch sections when year changes
  useEffect(() => {
    if (filters.dept_name && filters.year) {
      axios.get(`http://localhost:8000/sections/${filters.dept_name}/${filters.year}`)
        .then(response => setSections(response.data))
        .catch(error => console.error("Error fetching sections:", error));
      
      // Fetch subjects for the department and year
      axios.get(`http://localhost:8000/subjects/${filters.dept_name}/${filters.year}`)
        .then(response => setSubjects(response.data))
        .catch(error => console.error("Error fetching subjects:", error));
      
      // Reset section when year changes
      setFilters(prev => ({
        ...prev,
        section_name: "",
        subject_code: ""
      }));
    }
  }, [filters.dept_name, filters.year]);
  
  // Handle filter input changes
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters((prev) => ({ ...prev, [name]: value }));
  };
  
  // Handle date changes via DatePicker
  const handleDateChange = (date) => {
    setFilters(prev => ({
      ...prev,
      date: date
    }));
  };
  
  // Apply filters to fetch attendance data
  const handleApplyFilters = async () => {
    setLoading(true);
    setError(null);

    // Check if we have at least one filter
    if (!filters.dept_name && !filters.year && !filters.section_name && 
        !filters.subject_code && !filters.date) {
      setError("Please select at least one filter");
      setLoading(false);
      return;
    }

    // Prepare query parameters
    const params = new URLSearchParams();
    if (filters.dept_name) params.append("dept_name", filters.dept_name);
    if (filters.year) params.append("year", filters.year);
    if (filters.section_name) params.append("section_name", filters.section_name);
    if (filters.subject_code) params.append("subject_code", filters.subject_code);
    if (filters.date) {
      const formattedDate = filters.date.toLocaleDateString("en-US", {
        month: "2-digit",
        day: "2-digit",
        year: "numeric",
      });
      params.append("date", formattedDate);
    }

    try {
      // Use GET with URLSearchParams instead of POST
      const response = await axios.get(`http://localhost:8000/attendance?${params.toString()}`);
      
      if (!response.data || response.data.length === 0) {
        setError("No attendance records found for the selected filters");
        setAttendanceData([]);
        setGroupedData({});
      } else {
        setAttendanceData(response.data);
        groupAttendanceData(response.data);
      }
    } catch (err) {
      console.error("Error fetching attendance data:", err);
      setError(err.response?.data?.detail || "Failed to fetch attendance data. Please try different filters.");
      setAttendanceData([]);
      setGroupedData({});
    } finally {
      setLoading(false);
    }
  };
  
  // Reset filters
  const handleResetFilters = () => {
    setFilters({
      dept_name: "",
      year: "",
      section_name: "",
      subject_code: "",
      date: null,
    });
    setYears([]);
    setSections([]);
    setSubjects([]);
  };
  
  // Group and organize attendance data
  const groupAttendanceData = (data) => {
    const grouped = {};
    
    if (!Array.isArray(data) || data.length === 0) {
      setGroupedData({});
      return;
    }
    
    const sortedData = [...data].sort((a, b) => {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);
      return dateB - dateA;
    });
    
    sortedData.forEach(record => {
      const { date, year, dept_name, section_name, time_block, student_data } = record;
      
      if (!date || !year || !dept_name || !section_name || !time_block || !student_data) {
        console.warn("Invalid record format:", record);
        return;
      }
      
      if (!grouped[date]) {
        grouped[date] = {};
      }
      
      if (!grouped[date][year]) {
        grouped[date][year] = {};
      }
      
      if (!grouped[date][year][dept_name]) {
        grouped[date][year][dept_name] = {};
      }
      
      if (!grouped[date][year][dept_name][section_name]) {
        grouped[date][year][dept_name][section_name] = {
          timeBlocks: {},
          studentData: {}
        };
      }
      
      grouped[date][year][dept_name][section_name].timeBlocks[time_block.id] = {
        id: time_block.id,
        startTime: time_block.start_time,
        endTime: time_block.end_time
      };
      
      student_data.forEach(student => {
        if (!grouped[date][year][dept_name][section_name].studentData[student.student_id]) {
          grouped[date][year][dept_name][section_name].studentData[student.student_id] = {
            id: student.student_id,
            name: student.student_name,
            rollNo: student.roll_no,
            attendance: {}
          };
        }
        
        grouped[date][year][dept_name][section_name].studentData[student.student_id].attendance[time_block.id] = student.present;
      });
    });
    
    setGroupedData(grouped);
    
    const initialExpandedState = {};
    Object.keys(grouped).forEach(date => {
      initialExpandedState[date] = false;
      
      Object.keys(grouped[date] || {}).forEach(year => {
        initialExpandedState[`${date}-${year}`] = false;
        
        Object.keys(grouped[date][year] || {}).forEach(dept => {
          initialExpandedState[`${date}-${year}-${dept}`] = false;
          
          Object.keys(grouped[date][year][dept] || {}).forEach(section => {
            initialExpandedState[`${date}-${year}-${dept}-${section}`] = false;
          });
        });
      });
    });
    
    setExpandedGroups(initialExpandedState);
  };
  
  const toggleGroup = (groupId) => {
    setExpandedGroups(prev => ({
      ...prev,
      [groupId]: !prev[groupId]
    }));
  };
  
  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", { 
      weekday: 'short',
      year: 'numeric', 
      month: 'short', 
      day: 'numeric'
    });
  };
  
  const sortTimeBlocks = (blocks) => {
    return [...blocks].sort((a, b) => {
      const timeA = a.startTime.split(':').map(Number);
      const timeB = b.startTime.split(':').map(Number);
      if (timeA[0] === timeB[0]) return timeA[1] - timeB[1];
      return timeA[0] - timeB[0];
    });
  };
  
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="max-w-7xl mx-auto p-4 lg:p-6 mt-28 mb-16">
        <div className="bg-white rounded-xl shadow-lg p-4 md:p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-800">Attendance Records</h2>
            <button
              onClick={() => setIsFilterVisible(!isFilterVisible)}
              className="flex items-center text-blue-600 hover:text-blue-800 transition"
            >
              <Filter size={18} className="mr-1" />
              {isFilterVisible ? "Hide Filters" : "Show Filters"}
            </button>
          </div>
          
          {isFilterVisible && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
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
                      <option key={typeof dept === 'string' ? dept : dept.dept_name} 
                              value={typeof dept === 'string' ? dept : dept.dept_name}>
                        {typeof dept === 'string' ? dept : dept.dept_name}
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
                    {years.map((year) => (
                      <option key={typeof year === 'string' || typeof year === 'number' ? year : year.year} 
                              value={typeof year === 'string' || typeof year === 'number' ? year : year.year}>
                        {typeof year === 'string' || typeof year === 'number' ? year : year.year}
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
                    {sections.map((section) => (
                      <option key={typeof section === 'string' ? section : section.section_name || section.section_id} 
                              value={typeof section === 'string' ? section : section.section_name}>
                        {typeof section === 'string' ? section : section.section_name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Subject Code
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
                      <option key={subject.subject_code || subject.id} 
                              value={subject.subject_code}>
                        {subject.subject_name || subject.subject_code}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Date
                  </label>
                  <div className="relative">
                    <DatePicker
                      selected={filters.date}
                      onChange={handleDateChange}
                      className="w-full border border-gray-300 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500"
                      placeholderText="All Dates"
                      dateFormat="MM/dd/yyyy"
                      isClearable
                    />
                    <Calendar className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500" size={18} />
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap gap-2 justify-end">
                <button
                  onClick={handleResetFilters}
                  className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition"
                >
                  Reset
                </button>
                <button
                  onClick={handleApplyFilters}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                >
                  Apply Filters
                </button>
              </div>
            </>
          )}
        </div>

        <div className="bg-white rounded-xl shadow-lg p-4 md:p-6">
          {loading ? (
            <div className="flex justify-center p-10">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
          ) : error ? (
            <div className="text-red-600 p-4 text-center">{error}</div>
          ) : Object.keys(groupedData).length === 0 ? (
            <div className="text-gray-600 p-4 text-center">No attendance records found.</div>
          ) : (
            <div className="overflow-x-auto">
              <div className="space-y-6">
                {Object.keys(groupedData).map(date => (
                  <div key={date} className="border rounded-lg overflow-hidden">
                    <div 
                      className="bg-blue-50 p-3 cursor-pointer flex items-center"
                      onClick={() => toggleGroup(date)}
                    >
                      {expandedGroups[date] ? (
                        <ChevronDown size={20} className="text-blue-700 mr-2" />
                      ) : (
                        <ChevronRight size={20} className="text-blue-700 mr-2" />
                      )}
                      <h3 className="text-lg font-medium text-blue-800">
                        {formatDate(date)}
                      </h3>
                    </div>
                    
                    {expandedGroups[date] && (
                      <div className="p-2 space-y-4">
                        {Object.keys(groupedData[date])
                          .sort((a, b) => parseInt(a) - parseInt(b))
                          .map(year => (
                            <div key={`${date}-${year}`} className="border rounded-lg overflow-hidden ml-4">
                              <div 
                                className="bg-indigo-50 p-2 cursor-pointer flex items-center"
                                onClick={() => toggleGroup(`${date}-${year}`)}
                              >
                                {expandedGroups[`${date}-${year}`] ? (
                                  <ChevronDown size={18} className="text-indigo-700 mr-2" />
                                ) : (
                                  <ChevronRight size={18} className="text-indigo-700 mr-2" />
                                )}
                                <h4 className="font-medium text-indigo-800">
                                  Year {year}
                                </h4>
                              </div>
                              
                              {expandedGroups[`${date}-${year}`] && (
                                <div className="p-2 space-y-3">
                                  {Object.keys(groupedData[date][year]).map(dept => (
                                    <div key={`${date}-${year}-${dept}`} className="border rounded-lg overflow-hidden ml-4">
                                      <div 
                                        className="bg-purple-50 p-2 cursor-pointer flex items-center"
                                        onClick={() => toggleGroup(`${date}-${year}-${dept}`)}
                                      >
                                        {expandedGroups[`${date}-${year}-${dept}`] ? (
                                          <ChevronDown size={16} className="text-purple-700 mr-2" />
                                        ) : (
                                          <ChevronRight size={16} className="text-purple-700 mr-2" />
                                        )}
                                        <h5 className="font-medium text-purple-800">
                                          {dept} Department
                                        </h5>
                                      </div>
                                      
                                      {expandedGroups[`${date}-${year}-${dept}`] && (
                                        <div className="p-2 space-y-2">
                                          {Object.keys(groupedData[date][year][dept]).map(section => {
                                            const sectionData = groupedData[date][year][dept][section];
                                            const timeBlocksArray = Object.values(sectionData.timeBlocks);
                                            const sortedTimeBlocks = sortTimeBlocks(timeBlocksArray);
                                            
                                            const students = Object.values(sectionData.studentData)
                                              .sort((a, b) => a.rollNo.localeCompare(b.rollNo));
                                            
                                            return (
                                              <div key={`${date}-${year}-${dept}-${section}`} className="border rounded-lg overflow-hidden ml-4">
                                                <div 
                                                  className="bg-teal-50 p-2 cursor-pointer flex items-center"
                                                  onClick={() => toggleGroup(`${date}-${year}-${dept}-${section}`)}
                                                >
                                                  {expandedGroups[`${date}-${year}-${dept}-${section}`] ? (
                                                    <ChevronDown size={16} className="text-teal-700 mr-2" />
                                                  ) : (
                                                    <ChevronRight size={16} className="text-teal-700 mr-2" />
                                                  )}
                                                  <h5 className="font-medium text-teal-800">
                                                    Section {section}
                                                  </h5>
                                                </div>
                                                
                                                {expandedGroups[`${date}-${year}-${dept}-${section}`] && (
                                                  <div className="overflow-x-auto">
                                                    <table className="min-w-full divide-y divide-gray-200">
                                                      <thead className="bg-gray-50">
                                                        <tr>
                                                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Roll No</th>
                                                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Student Name</th>
                                                          {sortedTimeBlocks.map((block, idx) => (
                                                            <th key={`header-${block.id}-${idx}`} className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                              {block.startTime} - {block.endTime}
                                                            </th>
                                                          ))}
                                                        </tr>
                                                      </thead>
                                                      <tbody className="bg-white divide-y divide-gray-200">
                                                        {students.map((student) => (
                                                          <tr key={`student-${student.id}`}>
                                                            <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                                                              {student.rollNo}
                                                            </td>
                                                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-800">
                                                              {student.name}
                                                            </td>
                                                            {sortedTimeBlocks.map((block, idx) => (
                                                              <td key={`cell-${student.id}-${block.id}-${idx}`} className="px-4 py-3 whitespace-nowrap text-sm text-center">
                                                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                                  student.attendance[block.id] 
                                                                    ? 'bg-green-100 text-green-800' 
                                                                    : 'bg-red-100 text-red-800'
                                                                }`}>
                                                                  {student.attendance[block.id] ? 'Present' : 'Absent'}
                                                                </span>
                                                              </td>
                                                            ))}
                                                          </tr>
                                                        ))}
                                                      </tbody>
                                                    </table>
                                                  </div>
                                                )}
                                              </div>
                                            );
                                          })}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
      
      <Footer />
    </div>
  );
}