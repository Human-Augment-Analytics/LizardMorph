# Google Cloud deployment guide 
## Pre-req:
### Install gcloud cli
https://cloud.google.com/sdk/docs/install

### Login to gcloud
```
gcloud auth login
```

### Create a project
```
gcloud projects create lizard-xray
```
### Expose the environment variables (change the bucket name)
```
export PROJECT_ID=$(gcloud config get-value project)
export PROEJCT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
export BUCKET_NAME=lizard-x-ray
```

### Create a s3 bucket
```
gcloud storage buckets create gs://$BUCKET_NAME --location=us-east1 --enable-hierarchical-namespace
gcloud storage cp ./backend/better_predictor_auto.dat gs://$BUCKET_NAME
gcloud storage folders create --recursive gs://$BUCKET_NAME/sessions
```

## setup services for google cloud 
```
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/run.admin"
```

## deploy to google cloud (note you might need to set the environemnt variable again)
gcloud builds submit --config cloudbuild.yaml

## Command to get url 
```
gcloud run services describe lizard-xray --platform managed --format 'value(status.url)'
```
