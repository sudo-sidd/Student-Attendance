import './App.css'
import { useState } from 'react'
import { CloudArrowUpIcon, PhotoIcon, CheckCircleIcon } from '@heroicons/react/24/outline'
import { XCircleIcon } from '@heroicons/react/24/solid'
import Header from './components/Header'
import AttendanceTable from './components/AttendanceTable'

function App() {
  const [preview, setPreview] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [detectedStudentIds, setDetectedStudentIds] = useState([]);
  
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setUploadSuccess(false);
      setUploadError(null);
      setDetectedStudentIds([]);
    }
  };

  const uploadImage = async () => {
    if (!selectedImage) {
      setUploadError("Please select an image first.");
      return;
    }
  
    setIsUploading(true);
    setUploadError(null);
    
    const formData = new FormData();
    // Change from "image" to "file" to match what the API expects
    formData.append("file", selectedImage);
    // Add the threshold parameter which appears to be required by the API
    formData.append("threshold", "0.45");
  
    try {
      const response = await fetch("http://127.0.0.1:8000/upload-image/", {
        method: "POST",
        body: formData,
        // Don't set Content-Type header - the browser will set it correctly with the boundary
      });
  
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      setPreview(`data:image/jpeg;base64,${data.image_base64}`);

      console.log("Upload success:", data);
      
      // Extract face identities from the response format you showed
      if (data.faces && Array.isArray(data.faces)) {
        // Extract only identities that aren't "Unknown" and have sufficient confidence
        const detectedIds = data.faces
          .filter(face => face.identity !== "Unknown" && face.confidence > 0.45)
          .map(face => face.identity);
        
        setDetectedStudentIds(detectedIds);
        console.log("Detected student IDs:", detectedIds);
      } else {
        console.warn("No faces array found in response:", data);
        setDetectedStudentIds([]);
      }
      
      setUploadSuccess(true);
    } catch (error) {
      console.error("Upload failed:", error);
      setUploadError(error.message || "An unknown error occurred");
    } finally {
      setIsUploading(false);
    }
  };

  const resetImage = () => {
    setSelectedImage(null);
    setPreview(null);
    setUploadSuccess(false);
    setUploadError(null);
    setDetectedStudentIds([]);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header/>
      <main className="container mx-auto py-6 px-4">
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Classroom Student Recognition</h2>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
            <div className="flex flex-col items-center">
              <div className="h-64 w-full max-w-lg bg-gray-100 rounded-lg flex items-center justify-center mb-5 overflow-hidden">
                {preview ? (
                  <img src={preview} alt="Preview" className="max-h-full object-contain" />
                ) : (
                  <div className="text-center p-6">
                    <PhotoIcon className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                    <span className="text-gray-500">Upload a classroom image to detect attendance</span>
                  </div>
                )}
              </div>
              
              {uploadError && (
                <div className="w-full max-w-lg mb-4 bg-red-50 text-red-800 px-4 py-2 rounded-lg border border-red-200 flex items-center">
                  <XCircleIcon className="h-5 w-5 mr-2 text-red-600" />
                  <span>{uploadError}</span>
                </div>
              )}
              
              <div className="flex flex-wrap gap-3 justify-center">
                {!selectedImage && (
                  <label
                    htmlFor="image"
                    className="cursor-pointer inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg shadow hover:bg-blue-700 transition focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  >
                    <PhotoIcon className="h-5 w-5 mr-2" />
                    Choose Image
                    <input
                      type="file"
                      name="image"
                      id="image"
                      accept="image/*"
                      onChange={handleImageChange}
                      className="hidden"
                    />
                  </label>
                )}
                
                {selectedImage && !uploadSuccess && (
                  <>
                    <button
                      onClick={uploadImage}
                      disabled={isUploading}
                      className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-green-600 rounded-lg shadow hover:bg-green-700 transition focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-70 disabled:cursor-not-allowed"
                    >
                      {isUploading ? (
                        <>
                          <div className="animate-spin h-5 w-5 mr-2 border-t-2 border-b-2 border-white rounded-full"></div>
                          Processing...
                        </>
                      ) : (
                        <>
                          <CloudArrowUpIcon className="h-5 w-5 mr-2" />
                          Process Image
                        </>
                      )}
                    </button>
                    
                    <button
                      onClick={resetImage}
                      disabled={isUploading}
                      className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 rounded-lg shadow hover:bg-gray-300 transition focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 disabled:opacity-70 disabled:cursor-not-allowed"
                    >
                      Cancel
                    </button>
                  </>
                )}
                
                {uploadSuccess && (
                  <>
                    <div className="bg-green-50 text-green-800 px-4 py-2 rounded-lg border border-green-200 flex items-center">
                      <CheckCircleIcon className="h-5 w-5 mr-2" />
                      Image processed successfully! {detectedStudentIds.length > 0 ? `${detectedStudentIds.length} students detected.` : ''}
                    </div>
                    <button
                      onClick={resetImage}
                      className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 rounded-lg shadow hover:bg-gray-300 transition focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
                    >
                      Upload New Image
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        <AttendanceTable detectedStudentIds={detectedStudentIds} />
      </main>
    </div>
  )
}

export default App
