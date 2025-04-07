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
            className={`inline-flex items-center gap-x-2 px-4 py-2 text-sm font-semibold rounded-lg border 
                ${checked ? 'bg-green-600 text-white hover:bg-green-700' : 'bg-red-500 text-white border-red-600 hover:bg-red-700'} 
                focus:outline-none`}
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

    // Fetch available sections
    useEffect(() => {
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
            })
            .catch(error => console.error("Error fetching sections:", error));
    }, []);

    // Fetch records for the selected section
    useEffect(() => {
        if (section) {
            fetch(`http://127.0.0.1:8000/class/${section}`)
                .then(res => res.json())
                .then(data => setRecords(data))
                .catch(error => console.error("Error fetching records:", error));
        }
    }, [section]);

    const handleSectionChange = (e) => {
        setSection(e.target.value);
    };
    console.log(records)
    const generateRecord = () => {
        return records.map((record, index) => (
            <tr key={index} className="hover:bg-gray-300">
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
                    {/* {record.time} */}
                    "8:30 AM"
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-end text-sm font-medium">
                    <Button status={record.status} />
                </td>
            </tr>
        ));
    };

    return (
        <>
            <h1 className='text-center font-bold text-2xl mt-15'>AI & ML Attendance</h1>
            
            <div className="m-10">
                <div className="mb-6">
                    <label htmlFor="section-select" className="mr-3 font-medium">Select Section:</label>
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
        </>
    );
};

export default AttendanceTable;
