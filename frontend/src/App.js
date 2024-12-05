import React, { useState } from 'react';

function App() {
  const [inputImageUrl, setInputImageUrl] = useState(null);
  const [outputImageUrl, setOutputImageUrl] = useState(null);
  const [loading, setLoading] = useState(false); // To track loading state

  const handleUpload = async (e) => {
    const file = e.target.files[0];

    if (file) {
      // Show the input image immediately after selection
      const inputUrl = URL.createObjectURL(file);
      setInputImageUrl(inputUrl);

      const formData = new FormData();
      formData.append('image', file);

      setLoading(true); // Start loading spinner

      try {
        const response = await fetch('http://localhost:5000/upload', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const blob = await response.blob();
          const outputUrl = URL.createObjectURL(blob);
          setOutputImageUrl(outputUrl); // Set the output image URL
        } else {
          console.error('Failed to upload image');
        }
      } catch (err) {
        console.error('Error:', err);
      } finally {
        setLoading(false); // Stop loading spinner
      }
    }
  };

  return (
    <div style={{ textAlign: 'center', marginTop: '50px' }}>
      <h1>Automated Lizard X-Ray Landmarking App</h1>
      {/* Upload Button */}
      <label
        htmlFor="upload-button"
        style={{
          display: 'inline-block',
          padding: '10px 20px',
          backgroundColor: '#007BFF',
          color: 'white',
          borderRadius: '5px',
          cursor: 'pointer',
          fontSize: '16px',
        }}
      >
        Upload Image
      </label>
      <input
        id="upload-button"
        type="file"
        onChange={handleUpload}
        style={{ display: 'none' }} // Hide the default file input
      />

      {/* Display Images */}
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '30px' }}>
        {/* Input Image */}
        <div style={{ marginRight: '20px', textAlign: 'center' }}>
          {inputImageUrl && (
            <>
              <h3>Input Image</h3>
              <img
                src={inputImageUrl}
                alt="Input"
                style={{ maxWidth: '400px', maxHeight: '400px', border: '1px solid black' }}
              />
            </>
          )}
        </div>

        {/* Output Image or Loading Spinner */}
        <div style={{ marginLeft: '20px', textAlign: 'center' }}>
          {loading ? (
            <>
              <h3>Processing...</h3>
              <div
                style={{
                  border: '8px solid #f3f3f3',
                  borderTop: '8px solid #007BFF',
                  borderRadius: '50%',
                  width: '50px',
                  height: '50px',
                  animation: 'spin 1s linear infinite',
                  margin: '20px auto',
                }}
              />
              {/* Inline styles for spinner animation */}
              <style>
                {`
                  @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                  }
                `}
              </style>
            </>
          ) : (
            outputImageUrl && (
              <>
                <h3>Output Image</h3>
                <img
                  src={outputImageUrl}
                  alt="Output"
                  style={{ maxWidth: '400px', maxHeight: '400px', border: '1px solid black' }}
                />
              </>
            )
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
