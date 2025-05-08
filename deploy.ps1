# Deployment script for Full-Stack Lizard X-Ray Analysis App (PowerShell version)

# Colors for output
$GREEN = "Green"
$YELLOW = "Yellow"
$RED = "Red"

# Configuration variables - CHANGE THESE
$RESOURCE_GROUP = "lizard-app-rg"
$LOCATION = "eastus"
$CONTAINER_REGISTRY_NAME = "lizardcvregistry"  # Must be globally unique
$WEB_APP_NAME = "lizardcvapp"                  # Must be globally unique
$WEBAPP_PLAN = "lizard-app-plan"
$DOCKER_IMAGE_NAME = "lizardcv"
$DOCKER_IMAGE_TAG = "latest"

# Function to display section headers
function Show-Section {
    param (
        [string]$Title
    )
    Write-Host "`n========== $Title ==========`n" -ForegroundColor $YELLOW
}

# Check if the predictor file exists
function Test-Predictor {
    if (-not (Test-Path "backend/better_predictor_auto.dat")) {
        Write-Host "ERROR: The predictor file 'backend/better_predictor_auto.dat' is missing!" -ForegroundColor $RED
        Write-Host "Please download it from: https://gatech.box.com/s/qky0pu7hd3y0b8okvl3r7zgfiaj961vb"
        Write-Host "and place it in the backend directory."
        return $false
    }
    return $true
}

# Check dependencies
function Test-Dependencies {
    Show-Section "Checking Dependencies"
    
    $missingDeps = $false
    
    # Check for Python
    if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
        Write-Host "Python is not installed" -ForegroundColor $RED
        $missingDeps = $true
    } else {
        Write-Host "Python is installed" -ForegroundColor $GREEN
    }
    
    # Check for Node.js
    if (-not (Get-Command "node" -ErrorAction SilentlyContinue)) {
        Write-Host "Node.js is not installed" -ForegroundColor $RED
        $missingDeps = $true
    } else {
        Write-Host "Node.js is installed" -ForegroundColor $GREEN
    }
    
    # Check for Docker
    if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
        Write-Host "Docker is not installed" -ForegroundColor $RED
        $missingDeps = $true
    } else {
        Write-Host "Docker is installed" -ForegroundColor $GREEN
    }
    
    # Check for Azure CLI
    if (-not (Get-Command "az" -ErrorAction SilentlyContinue)) {
        Write-Host "Azure CLI is not installed" -ForegroundColor $RED
        Write-Host "Install Azure CLI from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        $missingDeps = $true
    } else {
        Write-Host "Azure CLI is installed" -ForegroundColor $GREEN
    }
    
    # Check directory structure
    if (-not (Test-Path "backend")) {
        Write-Host "Backend directory not found. Make sure you're in the project root directory." -ForegroundColor $RED
        $missingDeps = $true
    } else {
        Write-Host "Backend directory found" -ForegroundColor $GREEN
    }
    
    if (-not (Test-Path "frontend")) {
        Write-Host "Frontend directory not found. Make sure you're in the project root directory." -ForegroundColor $RED
        $missingDeps = $true
    } else {
        Write-Host "Frontend directory found" -ForegroundColor $GREEN
    }
    
    if ($missingDeps) {
        Write-Host "Please install the missing dependencies and try again." -ForegroundColor $RED
        return $false
    }
    
    return $true
}

# Check frontend package.json
function Test-FrontendPackage {
    if (-not (Test-Path "frontend/package.json")) {
        Write-Host "WARNING: frontend/package.json not found. Make sure your React app is set up correctly." -ForegroundColor $YELLOW
        return $false
    }
    
    # Check if build script exists in package.json
    $packageJson = Get-Content -Raw -Path "frontend/package.json" | ConvertFrom-Json
    
    if (-not $packageJson.scripts.build) {
        Write-Host "WARNING: No 'build' script found in package.json. Your React app might not build correctly." -ForegroundColor $YELLOW
        return $false
    }
    
    return $true
}

# Build and test Docker image locally
function Build-LocalDocker {
    Show-Section "Building and Testing Docker Image Locally"
    
    # First check the frontend package.json
    Test-FrontendPackage
    
    # Use the multi-stage Dockerfile in the project root
    if (-not (Test-Path "Dockerfile")) {
        Write-Host "Dockerfile not found in project root. Make sure you've created it as per instructions." -ForegroundColor $RED
        return $false
    }
    
    Write-Host "Building Docker image (this may take a while)..."
    docker build -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker build failed. Please check the errors above." -ForegroundColor $RED
        return $false
    }
    
    Write-Host "Docker image built successfully!" -ForegroundColor $GREEN
    
    # Run Docker container
    Write-Host "Starting Docker container for testing..."
    docker run -d -p 5000:5000 --name lizard-app-test "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to start Docker container. Please check the errors above." -ForegroundColor $RED
        return $false
    }
    
    Write-Host "Docker container started!" -ForegroundColor $GREEN
    Write-Host "The full app should be available at: http://localhost:5000" -ForegroundColor $GREEN
    Write-Host "The API endpoints should be available at: http://localhost:5000/data, /images, etc." -ForegroundColor $GREEN
    Write-Host "Press Enter to continue with deployment or Ctrl+C to stop..." -ForegroundColor $YELLOW
    Read-Host
    
    # Stop and remove test container
    Write-Host "Stopping and removing test container..."
    docker stop lizard-app-test
    docker rm lizard-app-test
    
    return $true
}

# Deploy to Azure
function Deploy-ToAzure {
    Show-Section "Deploying to Azure"
    
    # Login to Azure
    Write-Host "Logging in to Azure..."
    az login
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Azure login failed. Please try again." -ForegroundColor $RED
        return $false
    }
    
    # Create resource group
    Write-Host "Creating resource group: $RESOURCE_GROUP in $LOCATION..."
    az group create --name $RESOURCE_GROUP --location $LOCATION
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create resource group. Please check the errors above." -ForegroundColor $RED
        return $false
    }
    
    # Create Azure Container Registry
    Write-Host "Creating Azure Container Registry: $CONTAINER_REGISTRY_NAME..."
    az acr create --resource-group $RESOURCE_GROUP --name $CONTAINER_REGISTRY_NAME --sku Basic
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create Azure Container Registry. Please check the errors above." -ForegroundColor $RED
        return $false
    }
    
    # Build and push Docker image to ACR
    Write-Host "Building and pushing Docker image to Azure Container Registry (this may take a while)..."
    az acr build --resource-group $RESOURCE_GROUP --registry $CONTAINER_REGISTRY_NAME --image "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to build and push Docker image to ACR. Please check the errors above." -ForegroundColor $RED
        return $false
    }
    
    # Create App Service plan
    Write-Host "Creating App Service plan..."
    az appservice plan create --name $WEBAPP_PLAN --resource-group $RESOURCE_GROUP --sku B1 --is-linux
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create App Service plan. Please check the errors above." -ForegroundColor $RED
        return $false
    }
    
    # Store subscription ID for role assignment
    $SUBSCRIPTION_ID = az account show --query id --output tsv
    
    # Create web app and configure it to use the container image with managed identity
    Write-Host "Creating web app and configuring it to use the container image..."
    az webapp create `
        --resource-group $RESOURCE_GROUP `
        --plan $WEBAPP_PLAN `
        --name $WEB_APP_NAME `
        --assign-identity [system] `
        --role AcrPull `
        --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" `
        --acr-use-identity --acr-identity [system] `
        --container-image-name "$CONTAINER_REGISTRY_NAME.azurecr.io/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create web app. Please check the errors above." -ForegroundColor $RED
        return $false
    }
    
    # Configure app settings
    Write-Host "Configuring app settings..."
    az webapp config appsettings set `
        --resource-group $RESOURCE_GROUP `
        --name $WEB_APP_NAME `
        --settings WEBSITES_PORT=5000
    
    # Configure CORS for the web app (if needed)
    Write-Host "Configuring CORS..."
    az webapp cors add --allowed-origins "*" --resource-group $RESOURCE_GROUP --name $WEB_APP_NAME
    
    # Display deployment information
    Write-Host "`nDeployment completed successfully!" -ForegroundColor $GREEN
    Write-Host "Full App URL: https://$WEB_APP_NAME.azurewebsites.net" -ForegroundColor $YELLOW
    Write-Host "API Base URL: https://$WEB_APP_NAME.azurewebsites.net/data" -ForegroundColor $YELLOW
    Write-Host "Resource Group: $RESOURCE_GROUP" -ForegroundColor $YELLOW
    Write-Host "Container Registry: $CONTAINER_REGISTRY_NAME" -ForegroundColor $YELLOW
    
    return $true
}

# Clean up Azure resources
function Cleanup-Resources {
    Show-Section "Cleaning Up Azure Resources"
    
    Write-Host "WARNING: This will delete ALL resources in the resource group: $RESOURCE_GROUP" -ForegroundColor $YELLOW
    Write-Host "This action cannot be undone." -ForegroundColor $YELLOW
    $confirm = Read-Host "Are you sure you want to continue? (y/n)"
    
    if ($confirm -eq "y" -or $confirm -eq "Y") {
        Write-Host "Deleting resource group: $RESOURCE_GROUP..."
        az group delete --name $RESOURCE_GROUP --yes --no-wait
        
        Write-Host "Resource group deletion initiated. It may take a few minutes to complete." -ForegroundColor $GREEN
    } else {
        Write-Host "Cleanup canceled."
    }
}

# Main function
function Main {
    Write-Host "===== Lizard X-Ray Analysis App Full-Stack Deployment =====" -ForegroundColor $GREEN
    
    # Check for predictor file
    if (-not (Test-Predictor)) { return }
    
    # Check dependencies
    if (-not (Test-Dependencies)) { return }
    
    # Show menu
    do {
        Write-Host "`nSelect an option:`n" -ForegroundColor $YELLOW
        Write-Host "1: Build & Test Docker Locally"
        Write-Host "2: Deploy to Azure"
        Write-Host "3: Clean Up Azure Resources"
        Write-Host "4: Exit`n"
        
        $choice = Read-Host "Enter your choice (1-4)"
        
        switch ($choice) {
            "1" { Build-LocalDocker }
            "2" { Deploy-ToAzure }
            "3" { Cleanup-Resources }
            "4" { 
                Write-Host "Exiting..."
                return 
            }
            default { Write-Host "Invalid option: $choice" -ForegroundColor $RED }
        }
    } while ($true)
}

# Run main function
Main