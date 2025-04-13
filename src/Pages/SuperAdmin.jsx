import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function SuperAdmin() {
  const [activeTab, setActiveTab] = useState('departments');
  const [departments, setDepartments] = useState([]);
  const [batches, setBatches] = useState([]);
  const [sections, setSections] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Form states for manual adding
  const [deptName, setDeptName] = useState('');
  const [batchDept, setBatchDept] = useState('');
  const [batchYear, setBatchYear] = useState('');
  const [sectionBatch, setSectionBatch] = useState('');
  const [sectionName, setSectionName] = useState('');
  const [subjectCode, setSubjectCode] = useState('');
  const [subjectName, setSubjectName] = useState('');
  const [subjectDept, setSubjectDept] = useState('');
  const [subjectYear, setSubjectYear] = useState('');
  const [studentRegNum, setStudentRegNum] = useState('');
  const [studentName, setStudentName] = useState('');
  const [studentSection, setStudentSection] = useState('');

  // CSV upload states
  const [csvDept, setCsvDept] = useState('');
  const [csvYear, setCsvYear] = useState('');
  const [csvFile, setCsvFile] = useState(null);

  // Edit modal state
  const [editModal, setEditModal] = useState({ isOpen: false, type: '', data: null });

  // Fetch initial data
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [deptRes, batchRes, sectionRes, subjectRes, studentRes] = await Promise.all([
          axios.get('http://localhost:8000/departments'),
          axios.get('http://localhost:8000/batches'),
          axios.get('http://localhost:8000/sections'),
          axios.get('http://localhost:8000/subjects'),
          axios.get('http://localhost:8000/students'),
        ]);
        setDepartments(deptRes.data);
        setBatches(batchRes.data);
        setSections(sectionRes.data);
        setSubjects(subjectRes.data);
        setStudents(studentRes.data);
      } catch (error) {
        setError('Failed to load data: ' + (error.response?.data?.detail || error.message));
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // CRUD Handlers
  const handleAdd = async (e, type, payload) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`http://localhost:8000/${type}`, payload);
      if (type === 'departments') setDepartments([...departments, response.data]);
      if (type === 'batches') setBatches([...batches, response.data]);
      if (type === 'sections') setSections([...sections, response.data]);
      if (type === 'subjects') setSubjects([...subjects, response.data]);
      if (type === 'students') setStudents([...students, response.data]);
      resetForm(type);
      alert(`${type.slice(0, -1)} added successfully`);
    } catch (error) {
      setError('Failed to add: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = (type, data) => {
    setEditModal({ isOpen: true, type, data });
  };

  const handleUpdate = async (e) => {
    e.preventDefault();
    const { type, data } = editModal;
    setLoading(true);
    setError(null);
    try {
      let response;
      if (type === 'departments') {
        response = await axios.put(`http://localhost:8000/departments/${data.old_name}`, { dept_name: data.dept_name });
        setDepartments(departments.map(d => d.dept_name === data.old_name ? response.data : d));
      } else if (type === 'batches') {
        response = await axios.put(`http://localhost:8000/batches/${data.batch_id}`, {
          dept_name: data.dept_name,
          year: data.year,
        });
        setBatches(batches.map(b => b.batch_id === data.batch_id ? response.data : b));
      } else if (type === 'sections') {
        response = await axios.put(`http://localhost:8000/sections/${data.section_id}`, {
          batch_id: data.batch_id,
          section_name: data.section_name,
        });
        setSections(sections.map(s => s.section_id === data.section_id ? response.data : s));
      } else if (type === 'subjects') {
        response = await axios.put(`http://localhost:8000/subjects/${data.old_code}`, {
          subject_code: data.subject_code,
          subject_name: data.subject_name,
          dept_name: data.dept_name,
          year: data.year,
        });
        setSubjects(subjects.map(s => s.subject_code === data.old_code ? response.data : s));
      } else if (type === 'students') {
        response = await axios.put(`http://localhost:8000/students/${data.old_reg_num}`, {
          register_number: data.register_number,
          name: data.name,
          section_id: data.section_id,
        });
        setStudents(students.map(s => s.register_number === data.old_reg_num ? response.data : s));
      }
      setEditModal({ isOpen: false, type: '', data: null });
      alert(`${type.slice(0, -1)} updated successfully`);
    } catch (error) {
      setError('Failed to update: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (type, id) => {
    if (!confirm(`Are you sure you want to delete this ${type.slice(0, -1)}?`)) return;
    setLoading(true);
    setError(null);
    try {
      await axios.delete(`http://localhost:8000/${type}/${id}`);
      if (type === 'departments') setDepartments(departments.filter(d => d.dept_name !== id));
      if (type === 'batches') setBatches(batches.filter(b => b.batch_id !== id));
      if (type === 'sections') setSections(sections.filter(s => s.section_id !== id));
      if (type === 'subjects') setSubjects(subjects.filter(s => s.subject_code !== id));
      if (type === 'students') setStudents(students.filter(s => s.register_number !== id));
      alert(`${type.slice(0, -1)} deleted successfully`);
    } catch (error) {
      setError('Failed to delete: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleCsvUpload = async (e) => {
    e.preventDefault();
    if (!csvDept || !csvYear || !csvFile) {
      setError('Please select department, year, and upload a CSV file.');
      return;
    }
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', csvFile);
    formData.append('dept_name', csvDept);
    formData.append('year', parseInt(csvYear));

    try {
      const response = await axios.post('http://localhost:8000/upload-students-csv', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setStudents(response.data.students); // Update student list with upserted data
      setCsvDept('');
      setCsvYear('');
      setCsvFile(null);
      alert('Students uploaded successfully');
    } catch (error) {
      setError('Failed to upload CSV: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const resetForm = (type) => {
    if (type === 'departments') setDeptName('');
    if (type === 'batches') { setBatchDept(''); setBatchYear(''); }
    if (type === 'sections') { setSectionBatch(''); setSectionName(''); }
    if (type === 'subjects') { setSubjectCode(''); setSubjectName(''); setSubjectDept(''); setSubjectYear(''); }
    if (type === 'students') { setStudentRegNum(''); setStudentName(''); setStudentSection(''); }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-semibold text-gray-800 mb-6">Super Admin Panel</h1>

      {/* Loading and Error States */}
      {loading && (
        <div className="text-center mb-4">
          <svg className="animate-spin h-5 w-5 text-blue-600 inline-block" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8h8a8 8 0 01-16 0z" />
          </svg>
          <span className="ml-2 text-gray-600">Loading...</span>
        </div>
      )}
      {error && <div className="text-red-600 mb-4 text-center">{error}</div>}

      {/* Tabs */}
      <div className="flex gap-4 mb-6 border-b border-gray-200">
        {['departments', 'batches', 'sections', 'subjects', 'students'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 font-medium ${
              activeTab === tab ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-600 hover:text-blue-600'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="bg-white p-6 rounded-lg shadow-lg max-w-4xl mx-auto">
        {activeTab === 'departments' && (
          <>
            <form onSubmit={(e) => handleAdd(e, 'departments', { dept_name: deptName })} className="mb-6">
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <label className="block text-gray-700 font-medium mb-2">Department Name</label>
                  <input
                    type="text"
                    value={deptName}
                    onChange={(e) => setDeptName(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  />
                </div>
                <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400" disabled={loading}>
                  Add Department
                </button>
              </div>
            </form>
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-3 text-left text-gray-700">Name</th>
                  <th className="p-3 text-left text-gray-700">Actions</th>
                </tr>
              </thead>
              <tbody>
                {departments.map((dept) => (
                  <tr key={dept.dept_name} className="border-b">
                    <td className="p-3">{dept.dept_name}</td>
                    <td className="p-3">
                      <button
                        onClick={() => handleEdit('departments', { ...dept, old_name: dept.dept_name })}
                        className="px-3 py-1 bg-yellow-500 text-white rounded mr-2 hover:bg-yellow-600 disabled:bg-gray-400"
                        disabled={loading}
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDelete('departments', dept.dept_name)}
                        className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-400"
                        disabled={loading}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        {activeTab === 'batches' && (
          <>
            <form onSubmit={(e) => handleAdd(e, 'batches', { dept_name: batchDept, year: parseInt(batchYear) })} className="mb-6">
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <label className="block text-gray-700 font-medium mb-2">Department</label>
                  <select
                    value={batchDept}
                    onChange={(e) => setBatchDept(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  >
                    <option value="">Select Department</option>
                    {departments.map((dept) => (
                      <option key={dept.dept_name} value={dept.dept_name}>{dept.dept_name}</option>
                    ))}
                  </select>
                </div>
                <div className="flex-1">
                  <label className="block text-gray-700 font-medium mb-2">Year</label>
                  <input
                    type="number"
                    value={batchYear}
                    onChange={(e) => setBatchYear(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    min="1"
                    max="4"
                    required
                    disabled={loading}
                  />
                </div>
                <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400" disabled={loading}>
                  Add Batch
                </button>
              </div>
            </form>
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-3 text-left text-gray-700">Department</th>
                  <th className="p-3 text-left text-gray-700">Year</th>
                  <th className="p-3 text-left text-gray-700">Actions</th>
                </tr>
              </thead>
              <tbody>
                {batches.map((batch) => (
                  <tr key={batch.batch_id} className="border-b">
                    <td className="p-3">{batch.dept_name}</td>
                    <td className="p-3">{batch.year}</td>
                    <td className="p-3">
                      <button
                        onClick={() => handleEdit('batches', batch)}
                        className="px-3 py-1 bg-yellow-500 text-white rounded mr-2 hover:bg-yellow-600 disabled:bg-gray-400"
                        disabled={loading}
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDelete('batches', batch.batch_id)}
                        className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-400"
                        disabled={loading}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        {activeTab === 'sections' && (
          <>
            <form onSubmit={(e) => handleAdd(e, 'sections', { batch_id: parseInt(sectionBatch), section_name: sectionName })} className="mb-6">
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <label className="block text-gray-700 font-medium mb-2">Batch</label>
                  <select
                    value={sectionBatch}
                    onChange={(e) => setSectionBatch(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  >
                    <option value="">Select Batch</option>
                    {batches.map((batch) => (
                      <option key={batch.batch_id} value={batch.batch_id}>
                        {batch.dept_name} - Year {batch.year}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex-1">
                  <label className="block text-gray-700 font-medium mb-2">Section Name</label>
                  <input
                    type="text"
                    value={sectionName}
                    onChange={(e) => setSectionName(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  />
                </div>
                <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400" disabled={loading}>
                  Add Section
                </button>
              </div>
            </form>
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-3 text-left text-gray-700">Batch</th>
                  <th className="p-3 text-left text-gray-700">Section</th>
                  <th className="p-3 text-left text-gray-700">Actions</th>
                </tr>
              </thead>
              <tbody>
                {sections.map((section) => (
                  <tr key={section.section_id} className="border-b">
                    <td className="p-3">
                      {section.batch_id} ({batches.find(b => b.batch_id === section.batch_id)?.dept_name || 'N/A'} - Year {batches.find(b => b.batch_id === section.batch_id)?.year || 'N/A'})
                    </td>
                    <td className="p-3">{section.section_name}</td>
                    <td className="p-3">
                      <button
                        onClick={() => handleEdit('sections', section)}
                        className="px-3 py-1 bg-yellow-500 text-white rounded mr-2 hover:bg-yellow-600 disabled:bg-gray-400"
                        disabled={loading}
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDelete('sections', section.section_id)}
                        className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-400"
                        disabled={loading}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        {activeTab === 'subjects' && (
          <>
            <form onSubmit={(e) => handleAdd(e, 'subjects', { subject_code: subjectCode, subject_name: subjectName, dept_name: subjectDept, year: parseInt(subjectYear) })} className="mb-6">
              <div className="flex gap-4 items-end flex-wrap">
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-gray-700 font-medium mb-2">Subject Code</label>
                  <input
                    type="text"
                    value={subjectCode}
                    onChange={(e) => setSubjectCode(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  />
                </div>
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-gray-700 font-medium mb-2">Subject Name</label>
                  <input
                    type="text"
                    value={subjectName}
                    onChange={(e) => setSubjectName(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  />
                </div>
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-gray-700 font-medium mb-2">Department</label>
                  <select
                    value={subjectDept}
                    onChange={(e) => setSubjectDept(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  >
                    <option value="">Select Department</option>
                    {departments.map((dept) => (
                      <option key={dept.dept_name} value={dept.dept_name}>{dept.dept_name}</option>
                    ))}
                  </select>
                </div>
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-gray-700 font-medium mb-2">Year</label>
                  <input
                    type="number"
                    value={subjectYear}
                    onChange={(e) => setSubjectYear(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    min="1"
                    max="4"
                    required
                    disabled={loading}
                  />
                </div>
                <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400" disabled={loading}>
                  Add Subject
                </button>
              </div>
            </form>
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-3 text-left text-gray-700">Code</th>
                  <th className="p-3 text-left text-gray-700">Name</th>
                  <th className="p-3 text-left text-gray-700">Department</th>
                  <th className="p-3 text-left text-gray-700">Year</th>
                  <th className="p-3 text-left text-gray-700">Actions</th>
                </tr>
              </thead>
              <tbody>
                {subjects.map((subject) => (
                  <tr key={subject.subject_code} className="border-b">
                    <td className="p-3">{subject.subject_code}</td>
                    <td className="p-3">{subject.subject_name}</td>
                    <td className="p-3">{subject.dept_name}</td>
                    <td className="p-3">{subject.year}</td>
                    <td className="p-3">
                      <button
                        onClick={() => handleEdit('subjects', { ...subject, old_code: subject.subject_code })}
                        className="px-3 py-1 bg-yellow-500 text-white rounded mr-2 hover:bg-yellow-600 disabled:bg-gray-400"
                        disabled={loading}
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDelete('subjects', subject.subject_code)}
                        className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-400"
                        disabled={loading}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        {activeTab === 'students' && (
          <>
            {/* Manual Add Form */}
            <form onSubmit={(e) => handleAdd(e, 'students', { register_number: studentRegNum, name: studentName, section_id: parseInt(studentSection) })} className="mb-6">
              <div className="flex gap-4 items-end flex-wrap">
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-gray-700 font-medium mb-2">Register Number</label>
                  <input
                    type="text"
                    value={studentRegNum}
                    onChange={(e) => setStudentRegNum(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  />
                </div>
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-gray-700 font-medium mb-2">Name</label>
                  <input
                    type="text"
                    value={studentName}
                    onChange={(e) => setStudentName(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  />
                </div>
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-gray-700 font-medium mb-2">Section</label>
                  <select
                    value={studentSection}
                    onChange={(e) => setStudentSection(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  >
                    <option value="">Select Section</option>
                    {sections.map((section) => (
                      <option key={section.section_id} value={section.section_id}>
                        {section.section_name} (Batch {section.batch_id})
                      </option>
                    ))}
                  </select>
                </div>
                <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400" disabled={loading}>
                  Add Student
                </button>
              </div>
            </form>

            {/* CSV Upload Section */}
            <div className="mb-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-lg font-semibold mb-2">Upload Students via CSV</h3>
              <form onSubmit={handleCsvUpload} className="flex gap-4 items-end flex-wrap">
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-gray-700 font-medium mb-2">Department</label>
                  <select
                    value={csvDept}
                    onChange={(e) => setCsvDept(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  >
                    <option value="">Select Department</option>
                    {departments.map((dept) => (
                      <option key={dept.dept_name} value={dept.dept_name}>{dept.dept_name}</option>
                    ))}
                  </select>
                </div>
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-gray-700 font-medium mb-2">Year</label>
                  <select
                    value={csvYear}
                    onChange={(e) => setCsvYear(e.target.value)}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  >
                    <option value="">Select Year</option>
                    {[1, 2, 3, 4].map((year) => (
                      <option key={year} value={year}>{year}</option>
                    ))}
                  </select>
                </div>
                <div className="flex-1 min-w-[200px]">
                  <label className="block text-gray-700 font-medium mb-2">CSV File</label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setCsvFile(e.target.files[0])}
                    className="w-full p-2 border rounded-lg"
                    required
                    disabled={loading}
                  />
                </div>
                <button type="submit" className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400" disabled={loading}>
                  Upload CSV
                </button>
              </form>
              <div className="mt-4 text-sm text-gray-600">
                <p><strong>Notes:</strong></p>
                <ul className="list-disc pl-5">
                  <li>The CSV must contain the columns: <code>RegisterNumber</code>, <code>Name</code>, and <code>Section</code>.</li>
                  <li>Example: 
                    <pre className="bg-gray-200 p-2 rounded mt-1">
                      RegisterNumber,Name,Section<br/>
                      CS1A001,Alice,A<br/>
                      CS1A002,Bob,B
                    </pre>
                  </li>
                  <li><code>Section</code> must match existing section names for the selected department and year.</li>
                  <li>Existing students with matching <code>RegisterNumber</code> will be updated; new ones will be added.</li>
                </ul>
              </div>
            </div>

            {/* Students Table */}
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-3 text-left text-gray-700">Register Number</th>
                  <th className="p-3 text-left text-gray-700">Name</th>
                  <th className="p-3 text-left text-gray-700">Section</th>
                  <th className="p-3 text-left text-gray-700">Actions</th>
                </tr>
              </thead>
              <tbody>
                {students.map((student) => (
                  <tr key={student.register_number} className="border-b">
                    <td className="p-3">{student.register_number}</td>
                    <td className="p-3">{student.name}</td>
                    <td className="p-3">{student.section_name} (Batch {student.batch_id})</td>
                    <td className="p-3">
                      <button
                        onClick={() => handleEdit('students', { ...student, old_reg_num: student.register_number })}
                        className="px-3 py-1 bg-yellow-500 text-white rounded mr-2 hover:bg-yellow-600 disabled:bg-gray-400"
                        disabled={loading}
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDelete('students', student.register_number)}
                        className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-400"
                        disabled={loading}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>

      {/* Edit Modal */}
      {editModal.isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-md">
            <h2 className="text-xl font-semibold mb-4">Edit {editModal.type.slice(0, -1)}</h2>
            {error && <div className="text-red-600 mb-4">{error}</div>}
            <form onSubmit={handleUpdate}>
              {editModal.type === 'departments' && (
                <div className="mb-4">
                  <label className="block text-gray-700 font-medium mb-2">Department Name</label>
                  <input
                    type="text"
                    value={editModal.data.dept_name}
                    onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, dept_name: e.target.value } })}
                    className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                    required
                    disabled={loading}
                  />
                </div>
              )}
              {editModal.type === 'batches' && (
                <>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Department</label>
                    <select
                      value={editModal.data.dept_name}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, dept_name: e.target.value } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      required
                      disabled={loading}
                    >
                      <option value="">Select Department</option>
                      {departments.map((dept) => (
                        <option key={dept.dept_name} value={dept.dept_name}>{dept.dept_name}</option>
                      ))}
                    </select>
                  </div>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Year</label>
                    <input
                      type="number"
                      value={editModal.data.year}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, year: parseInt(e.target.value) } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      min="1"
                      max="4"
                      required
                      disabled={loading}
                    />
                  </div>
                </>
              )}
              {editModal.type === 'sections' && (
                <>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Batch</label>
                    <select
                      value={editModal.data.batch_id}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, batch_id: parseInt(e.target.value) } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      required
                      disabled={loading}
                    >
                      <option value="">Select Batch</option>
                      {batches.map((batch) => (
                        <option key={batch.batch_id} value={batch.batch_id}>
                          {batch.dept_name} - Year {batch.year}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Section Name</label>
                    <input
                      type="text"
                      value={editModal.data.section_name}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, section_name: e.target.value } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      required
                      disabled={loading}
                    />
                  </div>
                </>
              )}
              {editModal.type === 'subjects' && (
                <>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Subject Code</label>
                    <input
                      type="text"
                      value={editModal.data.subject_code}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, subject_code: e.target.value } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      required
                      disabled={loading}
                    />
                  </div>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Subject Name</label>
                    <input
                      type="text"
                      value={editModal.data.subject_name}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, subject_name: e.target.value } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      required
                      disabled={loading}
                    />
                  </div>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Department</label>
                    <select
                      value={editModal.data.dept_name}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, dept_name: e.target.value } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      required
                      disabled={loading}
                    >
                      <option value="">Select Department</option>
                      {departments.map((dept) => (
                        <option key={dept.dept_name} value={dept.dept_name}>{dept.dept_name}</option>
                      ))}
                    </select>
                  </div>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Year</label>
                    <input
                      type="number"
                      value={editModal.data.year}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, year: parseInt(e.target.value) } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      min="1"
                      max="4"
                      required
                      disabled={loading}
                    />
                  </div>
                </>
              )}
              {editModal.type === 'students' && (
                <>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Register Number</label>
                    <input
                      type="text"
                      value={editModal.data.register_number}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, register_number: e.target.value } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      required
                      disabled={loading}
                    />
                  </div>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Name</label>
                    <input
                      type="text"
                      value={editModal.data.name}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, name: e.target.value } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      required
                      disabled={loading}
                    />
                  </div>
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">Section</label>
                    <select
                      value={editModal.data.section_id}
                      onChange={(e) => setEditModal({ ...editModal, data: { ...editModal.data, section_id: parseInt(e.target.value) } })}
                      className="w-full p-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                      required
                      disabled={loading}
                    >
                      <option value="">Select Section</option>
                      {sections.map((section) => (
                        <option key={section.section_id} value={section.section_id}>
                          {section.section_name} (Batch {section.batch_id})
                        </option>
                      ))}
                    </select>
                  </div>
                </>
              )}
              <div className="flex gap-4">
                <button type="submit" className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400" disabled={loading}>
                  {loading ? 'Saving...' : 'Save'}
                </button>
                <button
                  type="button"
                  onClick={() => setEditModal({ isOpen: false, type: '', data: null })}
                  className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:bg-gray-400"
                  disabled={loading}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}