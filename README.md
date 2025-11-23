# React + Flask LizardMorph App
This app is built on the machine learning toolbox ml-morph. This app has a pre-trained model to predict 34 landmarks on lizard anole x-rays.

To learn more about the ml-morph toolbox: 

Porto, A. and Voje, K.L., 2020. ML‐morph: A fast, accurate and general approach for automated detection and landmarking of biological structures in images. Methods in Ecology and Evolution, 11(4), pp.500-512.

## Structure

- **Frontend**: Located in `frontend/`, built with React.
- **Backend**: Located in `backend/`, powered by Flask.

### Setup Instructions

#### Backend

Port availability may vary depending on your device. To specify a custom port for the backend, create a `.env` file in the root directory with the following content:
```
API_PORT=3000 # your desire port number
```
If no port is specified, the default is 5000.

1. Navigate to the `backend` folder.
2. Since the predictor is too big for this platform, download here: https://gatech.box.com/s/qky0pu7hd3y0b8okvl3r7zgfiaj961vb
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
   npm run dev
   ```

#### Setting up ngrok for External Access
If you want to make your local development server accessible from the internet:
https://ngrok.com/docs/getting-started/
1. Install ngrok:
   - Download from https://ngrok.com/download
   - Or `brew install ngrok` (macOS)

2. Configure ngrok:
   - Register a free account with ngrok 
   - Add your auth token
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```

3. Create/Update `.env` file (mentioned above) with the following
   ```
   VITE_ALLOWED_HOSTS=your-subdomain.ngrok-free.app
   ```
   Replace `your-subdomain` with your actual ngrok subdomain (e.g., "8e41-123-45-67-89.ngrok-free.app")

4. Start ngrok tunnel (after starting your frontend server):
   ```bash
   ngrok http 5173  # Use your frontend port number
   ```

5. Copy the generated ngrok URL
   and update your `.env` file with this domain.

Note: The ngrok URL will change each time you restart ngrok unless you have a paid account with a fixed subdomain.

#### Sample Environment file (.env)
```
API_PORT=3000 # your desire port number
VITE_ALLOWED_HOSTS=your-subdomain.ngrok-free.app
```

## Vignette
1. Open a terminal and activate the backend with the instructions from above
2. Open another terminal and activate the frontend with the instructions from above
3. Navigate to http://localhost:5000 
4. Hit upload on the webpage and select the picture from the folder sample_image in the project directory
5. Notice output.xml, output.tps, output.csv appear in project directory
6. Image should appear in the web browser:
   
![annotated_processed_June 1st 1_06-01-2024 10_18_37_1-1](https://github.com/user-attachments/assets/ad89d5f6-cfbf-4d17-acfe-1c1c1f2647cc)


7. Landmarks on image can be moved to fix predictions
8. Image can be viewed in three ways: original upload, color contrasted or inverted color
9. Image can be downloaded for records

## Docker Setup and Local Testing

To test the complete application locally using Docker:

1. Ensure Docker Desktop is installed and running
2. Make sure the predictor file (`better_predictor_auto.dat`) is in the `backend/` directory
3. Build and run the Docker container:

   ```bash
   # Build the Docker image
   docker build -t lizardcv:latest .

   # Run the container
   docker run -d -p 5000:5000 --name lizard-app-test lizardcv:latest
   ```

4. Access the application at http://localhost:5000
5. To stop and remove the container when done:
   ```bash
   docker stop lizard-app-test
   docker rm lizard-app-test
   ```

## Deployment Options

### Option 1: Running Locally (Non-Docker)

For development, you can run the frontend and backend separately:

1. Start the backend server:
   ```
   cd backend
   python app.py
   ```

2. In a separate terminal, start the frontend:
   ```
   cd frontend
   npm start
   ```

3. Access the frontend at http://localhost:3000

### Option 2: Automated Deployment with PowerShell Script

The project includes a PowerShell script (`deploy.ps1`) for simplified deployment:

1. Open PowerShell in the project root
2. Run the script:
   ```powershell
   .\deploy.ps1
   ```
3. Choose from the menu:
   - Option 1: Build & Test Docker Locally
   - Option 2: Deploy to Azure
   - Option 3: Clean Up Azure Resources

### Option 3: Manual Deployment to Azure

To deploy manually to Azure:

1. Login to Azure:
   ```
   az login
   ```

2. Create a resource group:
   ```
   az group create --name lizard-app-rg --location eastus
   ```

3. Create Azure Container Registry:
   ```
   az acr create --resource-group lizard-app-rg --name [your-registry-name] --sku Basic
   ```

4. Build and push the Docker image:
   ```
   az acr build --resource-group lizard-app-rg --registry [your-registry-name] --image lizardcv:latest .
   ```

5. Create App Service plan:
   ```
   az appservice plan create --name lizard-app-plan --resource-group lizard-app-rg --sku B1 --is-linux
   ```

6. Create and configure the web app:
   ```
   az webapp create --resource-group lizard-app-rg --plan lizard-app-plan --name [your-app-name] --deployment-container-image-name [your-registry-name].azurecr.io/lizardcv:latest
   ```

## ⚠️ Known Issues and Limitations

### Local Development and Deployment

1. **Predictor File**: The machine learning model file (`better_predictor_auto.dat`) is required but not included in the repository due to its size. You must download it separately and place it in the backend directory.

2. **Docker Memory Requirements**: The Docker container may require significant memory. If you encounter container crashes during build or runtime, try increasing Docker's allocated memory in settings.

### Azure Deployment Cautions

1. **Container Registry Authentication**: The current deployment script sets up managed identity for ACR Pull role, but this configuration may require additional permissions adjustments in some Azure environments.

2. **Processing Large Files**: The B1 App Service tier may be insufficient for processing multiple or large X-ray images. Consider upgrading to at least a B2 or P1v2 tier for production workloads.

3. **Gunicorn Configuration**: The current Gunicorn configuration in `gunicorn.conf.py` is set to bind to port 50505, but the Dockerfile and App Service expect port 5000. This discrepancy may cause connectivity issues when deployed to Azure. Modify the `gunicorn.conf.py` file to use port 5000 before deployment:

   ```python
   # In gunicorn.conf.py
   bind = "0.0.0.0:5000"  # Change from 50505 to 5000
   ```

4. **Cold Start Issues**: Azure App Service containers may experience cold start delays. The initial load of the application might take up to a minute due to the size of the ML models being loaded.

## Troubleshooting

- **Missing predictor file error**: Download the predictor file from the link above and place it in the backend directory.
- **Docker build failure**: Ensure Docker has enough allocated memory (at least 4GB recommended).
- **Frontend not loading**: Check browser console for CORS errors; ensure the backend API URL is correctly set.
- **Image processing errors**: Verify the uploaded image format is supported (JPG, PNG, TIF, BMP).
- **Azure deployment issues**: Check Azure logs with `az webapp log tail --resource-group lizard-app-rg --name [your-app-name]`




