# React + Flask LizardMorph App
This app is built on the machine learning toolbox ml-morph. This app has a pre-trained model to predict 34 landmarks on lizard anole x-rays.

To learn more about the ml-morph toolbox: 

Porto, A. and Voje, K.L., 2020. ML‐morph: A fast, accurate and general approach for automated detection and landmarking of biological structures in images. Methods in Ecology and Evolution, 11(4), pp.500-512.

## Structure

- **Frontend**: Located in `frontend/`, built with React.
- **Backend**: Located in `backend/`, powered by Flask.

### Setup Instructions

#### Backend
1. Navigate to the `backend` folder.
2. Since the predictor is too big for this platform, download here: https://gatech.box.com/s/ngg75ektk3zr2ed8085xa4cp3yjvm24q
3. Paste the predictor into the backend folder
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the Flask server:
   ```
   python app.py
   ```

#### Frontend
1. Install node.js and add it to the PATH.
2. Navigate to the `frontend` folder.
3. Install dependencies:
   ```
   npm install
   ```
4. Start the React app:
   ```
   npm start
   ```
## Vignette
1. Open a terminal and activate the backend with the instructions from above
2. Open another terminal and activate the frontend with the instructions from above
3. Navigate to http://localhost:5000 
4. Hit upload on the webpage and select the picture from the folder sample_image in the project directory
5. Notice output.xml, output.tps, output.csv appear in project directory
6. Image should appear in the web browser:
   
![annotated_processed_June 1st 1_06-01-2024 10_18_37_1-1](https://github.com/user-attachments/assets/ad89d5f6-cfbf-4d17-acfe-1c1c1f2647cc)



