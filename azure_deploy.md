## Azure 100$ Free Student

https://azure.microsoft.com/en-us/free/students


## Download Azure CLI 

https://aka.ms/installazurecliwindows

## Claude Says Do This and he's a Pretty Smart Guy


# Deployment Guide for Lizard X-Ray Analysis App

This guide covers how to deploy your full-stack Lizard X-Ray Analysis application with a React frontend and Flask backend to Azure App Service.

## Prerequisites

1. **Predictor File**: Download the predictor file from https://gatech.box.com/s/qky0pu7hd3y0b8okvl3r7zgfiaj961vb and place it in the `backend` directory as `better_predictor_auto.dat`

2. **Required Tools**:
   - Python 3.x
   - Node.js and npm
   - Docker Desktop
   - Azure CLI (for cloud deployment)

3. **Project Structure**:
   ```
   lizard-app/
   ├── frontend/           # React frontend
   │   ├── package.json
   │   ├── public/
   │   └── src/
   ├── backend/            # Flask backend
   │   ├── app.py
   │   ├── utils.py
   │   └── better_predictor_auto.dat
   ├── Dockerfile          # Multi-stage Dockerfile
   ├── .dockerignore
   └── deploy_fullstack.ps1
   ```

## Setup Instructions

1. **Update backend code to serve frontend**:
   - Add the route handlers from the "app.py Modifications" file to your backend's `app.py`
   - This ensures your Flask app can serve the React frontend's static files

2. **Prepare the Dockerfile**:
   - Copy the "Dockerfile (Multi-stage for React+Flask)" to your project root
   - This Dockerfile will:
     - Build your React frontend
     - Set up your Flask backend
     - Configure the app to serve both components

3. **Add .dockerignore file**:
   - Copy the provided `.dockerignore` file to your project root
   - This improves build performance and reduces image size

4. **Copy the PowerShell deployment script**:
   - Save `deploy_fullstack.ps1` to your project root
   - This script handles both local testing and Azure deployment

## Deployment Steps

### Local Testing

1. Open PowerShell in your project root
2. Run the deployment script:
   ```powershell
   .\deploy_fullstack.ps1
   ```
3. Select option 1 to build and test locally
4. After successful build, your app will be available at http://localhost:5000

### Deploy to Azure

1. Run the deployment script again
2. Select option 2 to deploy to Azure
3. Follow the prompts to log in to Azure and configure your deployment
4. Once deployed, your app will be available at https://[your-app-name].azurewebsites.net

## Troubleshooting

### Common Issues

1. **Frontend build fails**:
   - Check that your React app has a valid `build` script in `package.json`
   - Make sure all dependencies are installed

2. **Backend fails to start**:
   - Verify the predictor file is in the correct location
   - Check for missing Python dependencies

3. **Container fails to run**:
   - Check Docker logs: `docker logs lizard-app-test`
   - Verify port configurations match between Dockerfile and Flask app

### Getting Logs from Azure

To view logs from your deployed app:

```powershell
az webapp log tail --resource-group lizard-app-rg --name [your-app-name]
```

## Notes on Azure Deployment

- The deployment uses App Service with Web App for Containers
- The app uses a system-assigned managed identity to securely pull from Azure Container Registry
- Basic (B1) SKU is recommended as a minimum for this app due to memory requirements

## Cleanup

When you're done with the deployment:

1. Run the deployment script
2. Select option 3 to clean up Azure resources
3. Confirm the deletion (this will remove all created resources)



## Windows VmmemWSL Kill Command (Use Admin Terminal)
wsl --shutdown