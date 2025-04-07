import './App.css'
import { useState } from 'react'
import Header from './components/Header'
import AttendanceTable from './components/AttendanceTable'

function App() {
  const [preview, setPreview] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  
  const data = [
    {
      regno:"71402324078",
      name:"Sai",
      dept:"AI & ML",
      time: "8:30 AM",
      period: "Maths",
      status: true
    },
    {
      regno:"71402324088",
      name:"Sidd",
      dept:"AI & ML",
      time: "8:30 AM",
      period: "Maths",
      status: true
    },
    {
      regno:"71402324090",
      name:"Siva",
      dept:"AI & ML",
      time: "8:30 AM",
      period: "Maths",
      status: false
    },
    {
      regno:"71402324061",
      name:"Pavithiran",
      dept:"AI & ML",
      time: "8:30 AM",
      period: "Maths",
      status: true
    },
  ]
  

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const uploadImage = async () => {
    if (!selectedImage) {
      alert("Please select an image first.");
      return;
    }
  
    const formData = new FormData();
    formData.append("image", selectedImage); // "image" must match FastAPI param
  
    try {
      const response = await fetch("http://127.0.0.1:8000/upload-image/", {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
      console.log("Upload success:", data);
      alert("Upload success!");
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Upload failed");
    }
  };


  return (
    <>
      <Header/>
      <div className='p-4 text-center border-2 border-dashed border-gray-300 rounded-lg mx-10 my-5 overflow-x-hidden'>
        <div className="h-32 bg-gray-100 flex items-center justify-center mb-3 overflow-hidden">
          {preview ? (
            <img src={preview} alt="Preview" className="h-full object-contain " />
          ) : (
            <span className="text-gray-500">Preview image here</span>
          )}
        </div>
        <h2 className="mb-3 font-semibold">Upload classroom image</h2>
        <div className="flex justify-center">
          {selectedImage ? <button
            onClick={uploadImage}
            className="mt-4 px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition"
          >
            Upload to Server
          </button>: <label
            htmlFor="image"
            className="cursor-pointer inline-block px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg shadow hover:bg-blue-700 transition"
          >
            Choose Image
            <input
              type="file"
              name="image"
              id="image"
              accept="image/*"
              onChange={handleImageChange}
              className="hidden"
            />
          </label>}
        </div>
      </div>
      <AttendanceTable records={data}/>
    </>
  )
}

export default App
