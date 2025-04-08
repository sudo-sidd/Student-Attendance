import React, { useEffect, useState } from "react";
import { CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/solid';

const Button = ({ status }) => {
    const [checked, setChecked] = useState(status);

    useEffect(() => {
        setChecked(status);
    }, [status]);

    return (
        <button
            onClick={() => setChecked(prev => !prev)}
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

const AttendanceTable = () => {
    const [sections, setSections] = useState([]);
    const [section, setSection] = useState("");
    const [records, setRecords] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

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

    const handleSectionChange = (e) => {
        setSection(e.target.value);
    };

    const generateRecord = () => {
        return records.map((record, index) => (
            <tr key={index} className="hover:bg-gray-100 transition-colors duration-150">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800">
                    {record.RegisterNumber}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">
                    {record.FullName}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">
                    {record.Department}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">
                    8:30 AM
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-end text-sm font-medium">
                    <Button status={record.status} />
                </td>
            </tr>
        ));
    };

    const saveAttendance = () => {
        alert("Attendance data saved successfully!");
        // Implementation for saving attendance data would go here
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
            <h1 className='text-center font-bold text-2xl mb-6 text-gray-800'>Attendance Management</h1>
            
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
                    <button
                        onClick={saveAttendance}
                        className="px-4 py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                        Save Attendance
                    </button>
                </div>

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
