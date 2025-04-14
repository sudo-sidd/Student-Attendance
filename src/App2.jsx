import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Form from './Pages/AttendanceForm';

import AttendanceReview from './Pages/AttendanceReview';
import SuperAdmin from './Pages/SuperAdmin';
import AttendanceAssist from './Pages/AttendanceAssist';
import AttendanceReport from './Pages/AttendanceReport';

function App2() {
  return (
    <BrowserRouter>
        <Routes>
            <Route path='/' element={<Form />} />
            <Route path="/attendance-assist" element={<AttendanceAssist />} />
            <Route path="/review" element={<AttendanceReview />} />
            <Route path="/report" element={<AttendanceReport />} />
            <Route path="/superadmin" element={<SuperAdmin />} />
        </Routes>
    </BrowserRouter>
  )
}

export default App2
