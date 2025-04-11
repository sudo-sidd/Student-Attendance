import React, { useEffect, useState } from "react";
import { CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/solid';
import { ArrowDownTrayIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';

const Button = ({ status, onChange, recordId }) => {
    const [checked, setChecked] = useState(status);

    useEffect(() => {
        setChecked(status);
    }, [status]);

    const handleStatusChange = () => {
        const newStatus = !checked;
        setChecked(newStatus);
        
        if (onChange) {
            onChange(recordId, newStatus);
        }
    };

    return (
        <button
            onClick={handleStatusChange}
            className={`inline-flex items-center justify-center gap-x-2 px-4 py-2 text-sm font-semibold rounded-lg border transition-all duration-200 shadow-sm
                ${checked 
                    ? 'bg-green-600 text-white hover:bg-green-700 border-green-700' 
                    : 'bg-red-500 text-white border-red-600 hover:bg-red-700'} 
                focus:outline-none focus:ring-2 focus:ring-offset-2 ${checked ? 'focus:ring-green-500' : 'focus:ring-red-500'}`}
        >
            {checked ? <CheckCircleIcon className="w-4 h-4" /> : <XCircleIcon className="w-4 h-4" />}
            {checked ? 'Present' : 'Absent'}
        </button>
    );
};

const AttendanceTable = ({ detectedStudentIds = [] }) => {
    const [sections, setSections] = useState([]);
    const [section, setSection] = useState("");
    const [records, setRecords] = useState([]);
    const [filteredRecords, setFilteredRecords] = useState([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [attendanceData, setAttendanceData] = useState({});
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [saveSuccess, setSaveSuccess] = useState(false);
    const [error, setError] = useState(null);
    const [saveError, setSaveError] = useState(null);
    const [detectedIds, setDetectedIds] = useState([]);
    const [autoMarkedStudents, setAutoMarkedStudents] = useState([]);

    // Update detected IDs
    useEffect(() => {
        setDetectedIds(detectedStudentIds);
    }, [detectedStudentIds]);

    // Apply search filter whenever records or search query changes
    useEffect(() => {
        if (searchQuery.trim() === '') {
            setFilteredRecords(records);
        } else {
            const query = searchQuery.toLowerCase();
            const filtered = records.filter(record => {
                const regNumber = record.RegisterNumber ? String(record.RegisterNumber).toLowerCase() : "";
                const name = record.FullName ? record.FullName.toLowerCase() : "";
                const department = record.Department ? record.Department.toLowerCase() : "";
                
                return regNumber.includes(query) || 
                       name.includes(query) || 
                       department.includes(query);
            });
            setFilteredRecords(filtered);
        }
    }, [records, searchQuery]);

    // Fetch available sections
    useEffect(() => {
        setIsLoading(true);
        fetch("http://127.0.0.1:8000/sections", { headers: { "Accept": "application/json" } })
            .then(res => {
                if (!res.ok) {
                    throw new Error("Network response was not ok");
                }
                return res.json();
            })
            .then(data => {
                setSections(data.sections);
                if (data.sections.length > 0) {
                    setSection(data.sections[0]);
                }
                setIsLoading(false);
                setError(null);
            })
            .catch(error => {
                console.error("Error fetching sections:", error);
                setError("Failed to load sections. Please try again later.");
                setIsLoading(false);
            });
    }, []);

    // Fetch records for the selected section
    useEffect(() => {
        if (section) {
            setIsLoading(true);
            fetch(`http://127.0.0.1:8000/class/${section}`)
                .then(res => {
                    if (!res.ok) throw new Error("Failed to fetch records");
                    return res.json();
                })
                .then(data => {
                    setRecords(data);
                    setFilteredRecords(data);
                    
                    // Initialize attendance data object with current status
                    const initialAttendance = {};
                    data.forEach(record => {
                        initialAttendance[record.RegisterNumber] = {
                            status: record.status,
                            name: record.FullName,
                            department: record.Department
                        };
                    });
                    setAttendanceData(initialAttendance);
                    
                    setIsLoading(false);
                    setError(null);
                })
                .catch(error => {
                    console.error("Error fetching records:", error);
                    setError("Failed to load student records. Please try again later.");
                    setIsLoading(false);
                });
        }
    }, [section]);

    // Update detected IDs when props change and automatically mark students
    useEffect(() => {
        if (detectedStudentIds && Array.isArray(detectedStudentIds) && detectedStudentIds.length > 0) {
            console.log("Processing detected student IDs:", detectedStudentIds);
            setDetectedIds(detectedStudentIds);
            
            // Mark attendance based on detected IDs
            if (records.length > 0) {
            const newAutoMarkedStudents = [];
            const updatedAttendance = { ...attendanceData };
            
            records.forEach(record => {
                const regNumber = String(record.RegisterNumber);
                
                // Direct match: Compare detected IDs to regNumber directly
                const matches = detectedStudentIds.some(id => {
                return regNumber === String(id);
                });
                
                if (matches) {
                updatedAttendance[regNumber] = {
                    ...updatedAttendance[regNumber],
                    status: true
                };
                newAutoMarkedStudents.push(regNumber);
                }
            });
            
            if (newAutoMarkedStudents.length > 0) {
                setAttendanceData(updatedAttendance);
                setAutoMarkedStudents(newAutoMarkedStudents);
                console.log(`Automatically marked ${newAutoMarkedStudents.length} students as present:`, newAutoMarkedStudents);
            }
            }
        }
    }, [detectedStudentIds, records]);

    const handleSectionChange = (e) => {
        setSection(e.target.value);
        setSaveSuccess(false);
        setSaveError(null);
    };

    // Handle attendance status changes
    const handleAttendanceChange = (regNo, status) => {
        setAttendanceData(prev => ({
            ...prev,
            [regNo]: {
                ...prev[regNo],
                status: status
            }
        }));
        setSaveSuccess(false);
    };

    const generateRecord = () => {
        return filteredRecords.map((record, index) => {
            const regNo = record.RegisterNumber;
            const isAutoMarked = autoMarkedStudents.includes(regNo);
            
            return (
                <tr key={index} className={`hover:bg-gray-100 transition-colors duration-150 ${isAutoMarked ? 'bg-green-50' : ''}`}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800">
                        {regNo}
                        {isAutoMarked && (
                            <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                                Auto-detected
                            </span>
                        )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">
                        {record.FullName}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">
                        {/* {record.Department} */}
                        AI ML
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">
                        8:30 AM
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-end text-sm font-medium">
                        <Button 
                            status={attendanceData[regNo]?.status || record.status}
                            onChange={handleAttendanceChange}
                            recordId={regNo}
                        />
                    </td>
                </tr>
            );
        });
    };

    const saveAttendance = async () => {
        if (records.length === 0) {
            setSaveError("No attendance records to save");
            return;
        }

        setIsSaving(true);
        setSaveError(null);
        setSaveSuccess(false);

        // Create attendance data for API
        const today = new Date().toISOString().split('T')[0];
        const attendancePayload = {
            section: section,
            date: today,
            records: Object.entries(attendanceData).map(([regNo, data]) => ({
                RegisterNumber: regNo,
                FullName: data.name,
                Department: data.department,
                Status: data.status ? "Present" : "Absent"
            }))
        };

        try {
            const response = await fetch("http://127.0.0.1:8000/save-attendance-csv", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(attendancePayload)
            });

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }

            // For file download, get response as blob
            const blob = await response.blob();
            
            // Create a temporary URL for the blob
            const url = window.URL.createObjectURL(blob);
            
            // Create a link element to trigger download
            const a = document.createElement('a');
            
            // Get filename from Content-Disposition header if available
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = `attendance_${section}_${today.replace(/-/g, '')}.csv`;
            
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="(.+)"/);
                if (filenameMatch && filenameMatch[1]) {
                    filename = filenameMatch[1];
                }
            }
            
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            setSaveSuccess(true);
            setSaveError(null);
        } catch (error) {
            console.error("Error downloading attendance:", error);
            setSaveError(error.message || "Failed to download attendance data");
            setSaveSuccess(false);
        } finally {
            setIsSaving(false);
        }
    };

    if (error) {
        return (
            <div className="bg-red-50 border-l-4 border-red-500 p-4 m-10 rounded shadow">
                <div className="flex items-center">
                    <div className="flex-shrink-0">
                        <XCircleIcon className="h-5 w-5 text-red-500" />
                    </div>
                    <div className="ml-3">
                        <p className="text-sm text-red-700">{error}</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-white rounded-lg shadow-md p-6 mb-10">
            <h1 className='text-center font-bold text-2xl mb-6 text-gray-800'>AI & ML  Attendance</h1>
            
            <div className="mx-auto max-w-7xl">
                <div className="flex flex-wrap items-center justify-between mb-6 gap-4">
                    <div className="flex items-center">
                        <label htmlFor="section-select" className="mr-3 font-medium text-gray-700">Select Section:</label>
                        <select
                            id="section-select"
                            value={section}
                            onChange={handleSectionChange}
                            className="px-3 py-2 border rounded-lg"
                        >
                            {sections.map((sec, index) => (
                                <option key={index} value={sec}>
                                    {sec}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="flex items-center">
                        <label htmlFor="search-input" className="mr-3 font-medium text-gray-700">Search:</label>
                        <div className="relative">
                            <input
                                id="search-input"
                                type="text"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="px-3 py-2 border rounded-lg pl-10"
                                placeholder="Search by Reg No, Name, Dept"
                            />
                            <MagnifyingGlassIcon className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
                        </div>
                    </div>
                    <button
                        onClick={saveAttendance}
                        disabled={isLoading || isSaving}
                        className="inline-flex items-center px-4 py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-70 disabled:cursor-not-allowed"
                    >
                        {isSaving ? (
                            <>
                                <div className="animate-spin h-4 w-4 mr-2 border-t-2 border-b-2 border-white rounded-full"></div>
                                Generating CSV...
                            </>
                        ) : (
                            <>
                                <ArrowDownTrayIcon className="h-5 w-5 mr-2" />
                                Download CSV
                            </>
                        )}
                    </button>
                </div>

                {saveSuccess && (
                    <div className="bg-green-50 border-l-4 border-green-500 p-4 mb-6 rounded shadow">
                        <div className="flex items-center">
                            <div className="flex-shrink-0">
                                <CheckCircleIcon className="h-5 w-5 text-green-500" />
                            </div>
                            <div className="ml-3">
                                <p className="text-sm text-green-700">Attendance data downloaded successfully!</p>
                            </div>
                        </div>
                    </div>
                )}

                {saveError && (
                    <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded shadow">
                        <div className="flex items-center">
                            <div className="flex-shrink-0">
                                <XCircleIcon className="h-5 w-5 text-red-500" />
                            </div>
                            <div className="ml-3">
                                <p className="text-sm text-red-700">{saveError}</p>
                            </div>
                        </div>
                    </div>
                )}

                {autoMarkedStudents.length > 0 && (
                    <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6 rounded shadow">
                        <div className="flex items-center">
                            <div className="flex-shrink-0">
                                <CheckCircleIcon className="h-5 w-5 text-blue-500" />
                            </div>
                            <div className="ml-3">
                                <p className="text-sm text-blue-700">
                                    <strong>{autoMarkedStudents.length}</strong> students were automatically marked present based on facial recognition
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                <div className="flex flex-col">
                    <div className="-m-1.5 overflow-auto">
                        <div className="p-1.5 min-w-full inline-block align-middle">
                            <div className="overflow-hidden">
                                <table className="min-w-full divide-y divide-gray-200 border-gray-500 border-2">
                                    <thead>
                                        <tr className="border-2 border-gray-600">
                                            <th className="px-6 py-3 text-start text-xs font-medium text-gray-500 uppercase">
                                                Reg No
                                            </th>
                                            <th className="px-6 py-3 text-start text-xs font-medium text-gray-500 uppercase">
                                                Name
                                            </th>
                                            <th className="px-6 py-3 text-start text-xs font-medium text-gray-500 uppercase">
                                                Department
                                            </th>
                                            <th className="px-6 py-3 text-end text-xs font-medium text-gray-500 uppercase">
                                                Time
                                            </th>
                                            <th className="px-6 py-3 text-end text-xs font-medium text-gray-500 uppercase">
                                                Present / Absent
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-200">
                                        {generateRecord()}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AttendanceTable;
