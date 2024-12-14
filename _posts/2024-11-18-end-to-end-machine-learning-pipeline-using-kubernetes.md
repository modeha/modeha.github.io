---
layout: post
title: End-to-End Machine Learning Pipeline Using Kubernetes
date: 2024-11-18 12:26 -0500
---
 **End-to-End Machine Learning Pipeline** using Kubernetes, starting from the dataset to deploying a trained model. 
 
 Here's the workflow:


### **Setup Overview**
We'll use Kubernetes to:
1. Preprocess a dataset.
2. Train a model using `train.py`.
3. Save the trained model.
4. Deploy the trained model as an API for predictions.

---

### **Prerequisites**
1. **Install Kubernetes on your Mac**:
   - Use **Docker Desktop** with Kubernetes enabled, or install Kubernetes via **Minikube**.
2. **Install `kubectl`**:
   - Verify Kubernetes is running:
     ```bash
     kubectl get nodes
     ```
3. **Install Python** (if needed) and ML libraries like `scikit-learn` or `TensorFlow`.
4. **Install Helm** (optional): For managing Kubernetes packages.

---

### **Step 1: Dataset Preparation**
We'll use a simple CSV dataset for house prices:
```csv
# Save this as dataset.csv
square_footage,bedrooms,bathrooms,price
1400,3,2,300000
1600,4,2,350000
1700,4,3,400000
1200,2,1,200000
1500,3,2,320000
```

Place this dataset in a directory, for example, `/Users/yourname/k8s-ml-pipeline`.

---

### **Step 2: Create a `train.py` Script**
Here’s a basic training script using `scikit-learn`:

```python
# train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv("dataset.csv")

# Features and target variable
X = data[["square_footage", "bedrooms", "bathrooms"]]
y = data["price"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
```

---

### **Step 3: Dockerize `train.py`**
1. **Create a `Dockerfile`:**
   ```dockerfile
   FROM python:3.9-slim

   # Copy files into the container
   COPY train.py /app/train.py
   COPY dataset.csv /app/dataset.csv

   # Set the working directory
   WORKDIR /app

   # Install dependencies
   RUN pip install pandas scikit-learn

   # Default command
   CMD ["python", "train.py"]
   ```

2. **Build the Docker Image**:
   ```bash
   docker build -t train-ml:latest .
   ```

---

### **Step 4: Create a Kubernetes Job for Training**
1. **Job YAML** (`train-job.yaml`):
   ```yaml
   apiVersion: batch/v1
   kind: Job
   metadata:
     name: train-job
   spec:
     template:
       spec:
         containers:
         - name: train-container
           image: train-ml:latest
           volumeMounts:
           - mountPath: /app
             name: model-volume
         restartPolicy: Never
         volumes:
         - name: model-volume
           hostPath:
             path: /Users/yourname/k8s-ml-pipeline
   ```

2. **Run the Job**:
   ```bash
   kubectl apply -f train-job.yaml
   ```

3. **Check Logs**:
   ```bash
   kubectl logs job/train-job
   ```

   This will output:
   ```
   Model trained and saved as model.pkl
   ```

The `model.pkl` file will be saved locally in `/Users/yourname/k8s-ml-pipeline`.

---

### **Step 5: Deploy the Trained Model as an API**
1. **Create a `predict.py` Script**:
   ```python
   # predict.py
   import pickle
   from flask import Flask, request, jsonify

   # Load the trained model
   with open("model.pkl", "rb") as f:
       model = pickle.load(f)

   app = Flask(__name__)

   @app.route("/predict", methods=["POST"])
   def predict():
       data = request.get_json()
       X = [[data["square_footage"], data["bedrooms"], data["bathrooms"]]]
       prediction = model.predict(X)
       return jsonify({"predicted_price": prediction[0]})

   if __name__ == "__main__":
       app.run(host="0.0.0.0", port=5000)
   ```

2. **Dockerize `predict.py`**:
   ```dockerfile
   FROM python:3.9-slim

   # Copy files
   COPY predict.py /app/predict.py
   COPY model.pkl /app/model.pkl

   # Set working directory
   WORKDIR /app

   # Install dependencies
   RUN pip install flask scikit-learn

   # Default command
   CMD ["python", "predict.py"]
   ```

3. **Build the API Docker Image**:
   ```bash
   docker build -t predict-ml:latest .
   ```

4. **Deployment YAML** (`predict-deployment.yaml`):
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: predict-api
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: predict-api
     template:
       metadata:
         labels:
           app: predict-api
       spec:
         containers:
         - name: predict-container
           image: predict-ml:latest
           ports:
           - containerPort: 5000
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: predict-service
   spec:
     selector:
       app: predict-api
     ports:
       - protocol: TCP
         port: 80
         targetPort: 5000
     type: LoadBalancer
   ```

5. **Deploy the API**:
   ```bash
   kubectl apply -f predict-deployment.yaml
   ```

6. **Access the API**:
   - Find the service IP:
     ```bash
     kubectl get services
     ```
   - Test the API:
     ```bash
     curl -X POST -H "Content-Type: application/json" \
       -d '{"square_footage": 1600, "bedrooms": 3, "bathrooms": 2}' \
       http://<EXTERNAL-IP>/predict
     ```

---

### **Step 6: Clean Up**
To clean up resources:
```bash
kubectl delete -f train-job.yaml
kubectl delete -f predict-deployment.yaml
```

---

### **Summary**
1. **Dataset**: Prepared and mounted into the container.
2. **Training**: Kubernetes Job ran `train.py` and saved the model.
3. **API Deployment**: The trained model was deployed as a REST API using Kubernetes Deployment and Service.

This pipeline can scale as needed and is fully containerized for portability and reproducibility.