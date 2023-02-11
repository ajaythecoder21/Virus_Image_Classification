# Virus_Image_Classification

## Goal
The goal of this project is to predict whether a patient has Viral Pneumonia, COVID-19, or is normal through Chest X-Ray Images.

## Setup the Data in Google Colab
-  The Images Dataset can be found on Kaggle at this link: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset 
-  After downloading the dataset, upload the dataset to Google Drive

## Tools/Libraries Used
-  Python General Libraries (pathlib, os, zipfile, random, PIL, io)
- PyTorch (torch, torchvision)
- Web Framework (Flask)
- Devops ML Containerization (Docker)
- Front-End (HTML and Bootstrap)

## Instructions
1. Run the Virus_Image_Classification.ipynb file for Deep Dive into the Deep Learning Aspects of this Project
2. The Notebook will have Sections to Follow Along
3. Please Install Docker and create a Docker Hub account
4. Run the following commands:
    - docker compose up --build
    - docker login
    - docker tag <source_image_name> <dest_image_name>:<version_name>
    - docker push <image_name>:<version_name>
    - docker run -p<port_number>:<port_number> <image_name>:<version_name>
5. Can View the Application Locally in the Browser

## Resources
https://www.learnpytorch.io
https://www.youtube.com/watch?v=zGP_nYmZd9c&t=0s
