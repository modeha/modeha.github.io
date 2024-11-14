---
layout: post
title: End-to-End Deployment of an AI Model Using Docker, Azure, and Streamlit
date: 2024-11-19 07:24 -0500
---


### **section 4: End-to-End Deployment of an AI Model Using Docker, Azure, and Streamlit**

---

#### **4.1 Designing the AI Solution**

   - **Overview of the AI Model Pipeline**
     - The pipeline for deploying an AI model typically includes stages like data ingestion, preprocessing, model inference, and visualization. In this section, we’ll walk through deploying an image classification model as a web application using Docker, Azure, and Streamlit.
     - **Pipeline Steps**:
       - **Input Handling**: The app will allow users to upload an image.
       - **Data Preprocessing**: Image resizing and scaling for compatibility with the model.
       - **Model Inference**: Running the model to get predictions.
       - **Output Visualization**: Displaying the prediction results in a user-friendly interface.

   - **High-Level Architecture**
     - The solution’s architecture includes the following components:
       - **Streamlit Front-End**: The user-facing interface, where users upload images and see predictions.
       - **Dockerized Application**: Encapsulates the model and application code in a Docker container for consistency across environments.
       - **Azure Cloud Platform**: Hosts the Dockerized application, making it accessible as a web service.

---

#### **4.2 Preparing the Docker Container**

   - **Writing the Dockerfile**
     - The Dockerfile serves as the blueprint for creating a container that includes all dependencies for running the Streamlit application and model.
     - Sample Dockerfile for an AI application:
       ```Dockerfile
       # Start with a base Python image
       FROM python:3.8

       # Set the working directory
       WORKDIR /app

       # Copy the current directory contents into the container
       COPY . /app

       # Install dependencies
       RUN pip install -r requirements.txt

       # Expose the port on which Streamlit will run
       EXPOSE 8501

       # Run the application
       CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
       ```
       - **Explanation**:
         - **`FROM python:3.8`**: Specifies the base image.
         - **`WORKDIR /app`** and **`COPY . /app`**: Sets the working directory and copies the local files.
         - **`RUN pip install -r requirements.txt`**: Installs required packages (e.g., Streamlit, TensorFlow, PyTorch).
         - **`EXPOSE 8501`**: Exposes the default Streamlit port.
         - **`CMD [...]`**: Runs the Streamlit app when the container starts.

   - **Building the Docker Image**
     - After defining the Dockerfile, build the Docker image:
       ```bash
       docker build -t ai-streamlit-app .
       ```
     - This command packages the code, dependencies, and environment into a Docker image named `ai-streamlit-app`.

---

#### **4.3 Deploying the Docker Container on Azure**

   - **Azure Container Instances (ACI) for Simple Deployments**
     - **Push the Docker Image to Azure Container Registry (ACR)**:
       1. First, create a container registry in Azure.
          ```bash
          az acr create --resource-group myResourceGroup --name myContainerRegistry --sku Basic
          ```
       2. Log in to the registry and push the Docker image:
          ```bash
          az acr login --name myContainerRegistry
          docker tag ai-streamlit-app myContainerRegistry.azurecr.io/ai-streamlit-app
          docker push myContainerRegistry.azurecr.io/ai-streamlit-app
          ```

     - **Deploy the Image to ACI**:
       - Create a container instance in ACI:
         ```bash
         az container create \
           --resource-group myResourceGroup \
           --name aiAppInstance \
           --image myContainerRegistry.azurecr.io/ai-streamlit-app \
           --cpu 1 --memory 1 \
           --registry-login-server myContainerRegistry.azurecr.io \
           --registry-username <username> \
           --registry-password <password> \
           --dns-name-label ai-streamlit-app \
           --ports 8501
         ```
       - **Accessing the Deployed App**:
         - The deployed application is now accessible at `http://ai-streamlit-app.region.azurecontainer.io:8501`.

   - **Azure Kubernetes Service (AKS) for Scalable Deployments**
     - **Why Use AKS?**: AKS provides orchestration for managing multiple containers, load balancing, and scaling.
     - **Deploying on AKS**:
       - Create an AKS cluster and configure it to pull images from ACR, providing a more robust and scalable deployment option for production environments.

---

#### **4.4 Building and Linking the Streamlit Front-End**

   - **Creating the Streamlit Application Code (`app.py`)**
     - Below is a sample Streamlit application to handle image uploads, preprocess the images, and display model predictions.
       ```python
       import streamlit as st
       from PIL import Image
       import tensorflow as tf

       # Load the model
       model = tf.keras.models.load_model("my_model.h5")

       # App title and instructions
       st.title("Image Classification App")
       st.write("Upload an image to classify.")

       # File uploader widget
       uploaded_file = st.file_uploader("Choose an image...", type="jpg")
       if uploaded_file is not None:
           image = Image.open(uploaded_file)
           st.image(image, caption="Uploaded Image", use_column_width=True)

           if st.button("Classify Image"):
               # Preprocess image
               image = image.resize((224, 224))
               image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
               image = image.reshape((1, 224, 224, 3))

               # Predict
               predictions = model.predict(image)
               st.write(f"Predicted class: {predictions.argmax()}")
       ```

   - **Testing the Application Locally**
     - Run the Streamlit app locally using Docker:
       ```bash
       docker run -p 8501:8501 ai-streamlit-app
       ```
     - Access the app at `http://localhost:8501` to verify functionality before deploying.

---

#### **4.5 Monitoring, Scaling, and Updating the Model**

   - **Monitoring Model Performance with Azure Monitor**
     - Azure Monitor collects logs and metrics for deployed applications, providing insights into model usage, prediction times, and errors.
     - Integrate Azure Monitor with ACI or AKS to capture logs from the container instances.

   - **Scaling the Application**
     - In AKS, configure the **Horizontal Pod Autoscaler (HPA)** to automatically scale the number of replicas based on CPU or memory utilization, ensuring high availability.
     - Example HPA configuration:
       ```yaml
       apiVersion: autoscaling/v1
       kind: HorizontalPodAutoscaler
       metadata:
         name: ai-streamlit-app
       spec:
         maxReplicas: 10
         minReplicas: 1
         targetCPUUtilizationPercentage: 50
       ```

   - **Updating the Model and Redeploying**
     - Update the model, rebuild the Docker image, and push it to ACR. Use the following commands:
       ```bash
       docker build -t ai-streamlit-app .
       docker tag ai-streamlit-app myContainerRegistry.azurecr.io/ai-streamlit-app
       docker push myContainerRegistry.azurecr.io/ai-streamlit-app
       ```
     - Deploy the updated image in ACI or AKS to apply changes to the live application.

---

#### **4.6 Implementing Continuous Integration/Continuous Deployment (CI/CD) with Azure DevOps**

   - **Setting Up Azure DevOps Pipelines**
     - Azure DevOps allows automated building, testing, and deployment of Docker images.
     - **Example YAML Pipeline**:
       ```yaml
       trigger:
         branches:
           include:
             - main

       pool:
         vmImage: 'ubuntu-latest'

       steps:
       - task: Docker@2
         inputs:
           containerRegistry: 'myContainerRegistry'
           repository: 'ai-streamlit-app'
           command: 'buildAndPush'
           tags: '$(Build.BuildId)'

       - task: AzureCLI@2
         inputs:
           azureSubscription: '<Your Subscription>'
           scriptType: 'bash'
           scriptLocation: 'inlineScript'
           inlineScript: |
             az container create --resource-group myResourceGroup --name aiAppInstance --image myContainerRegistry.azurecr.io/ai-streamlit-app:$(Build.BuildId) --cpu 1 --memory 1 --dns-name-label ai-streamlit-app --ports 8501
       ```

   - **Automating Updates and Monitoring CI/CD Pipeline**
     - Each code push triggers the pipeline to rebuild the Docker image, push it to ACR, and deploy the updated container.
     - This setup allows rapid iteration and updates, ensuring the deployed AI model remains current with minimal manual intervention.

---

#### **4.7 Best Practices and Final Thoughts**

   - **Security and Access Control**
     - Restrict access to ACR, ACI, and AKS resources by configuring role-based access control (RBAC).
     - Use **Azure Key Vault** for secure storage of sensitive data like API keys and database credentials.

   - **Optimizing Costs and Resources**
     - Monitor and analyze usage to optimize resource allocation and cost-effectiveness, especially when scaling up in AKS.
     - Enable auto-scaling