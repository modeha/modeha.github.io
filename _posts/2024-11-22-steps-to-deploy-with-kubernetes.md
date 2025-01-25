---
layout: post
title: Steps to Deploy with Kubernetes
date: 2024-11-22 10:52 -0500
---
To provide a detailed step-by-step explanation of Kubernetes in the context of the files you've uploaded, I'll use your project as an example. Here's an overview and breakdown:

---

### 1. **What is Kubernetes?**
Kubernetes is an open-source container orchestration platform for managing containerized applications across a cluster of machines. It automates deployment, scaling, and operations of application containers.

**Key Concepts:**
- **Pods**: Smallest deployable units, encapsulating one or more containers.
- **Services**: Expose a set of Pods as a network service.
- **Deployments**: Define how Pods are created and managed.
- **Jobs**: Ensure specific tasks are completed (e.g., model training).
- **Volumes**: Provide persistent storage for containers.

---

### 2. **Overview of Your Project**
Your project involves:
- **Flask API**: `predict.py` serves predictions.
- **Streamlit App**: `app.py` interacts with users to send requests to the API.
- **Model Training**: `train.py` trains and saves a linear regression model.
- **Kubernetes Deployment**: Managed using YAML files (`deployment.yaml`, `service.yaml`, `train-job.yaml`) and `run_pipeline.sh`.

---

### 3. **Steps to Deploy with Kubernetes**

#### **Step 1: Containerize the Application**
Kubernetes uses Docker containers. Your `Dockerfile` ensures:
1. The environment is consistent.
2. Dependencies for `predict.py` are installed.
3. The application is runnable.

**Example Dockerfile** (assumed from context):
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "predict.py"]
```

---

#### **Step 2: Kubernetes Job for Training**
Your `run_pipeline.sh` creates a Kubernetes Job to train the model.

**Key Steps in Training Job**:
- Volume mounts provide the dataset (`dataset.csv`) and a path to save `model.pkl`.
- Job YAML dynamically applies training logic using `train.py`.

**Snippet from `run_pipeline.sh`**:
```bash
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: train-job
spec:
  template:
    spec:
      containers:
      - name: train-job
        image: $DOCKER_IMAGE
        command: ["python", "train.py"]
      volumes:
      - name: dataset-volume
        hostPath:
          path: /mnt/data/dataset.csv
EOF
```

---

#### **Step 3: API Deployment**
After training, the Flask API (`predict.py`) is deployed. Kubernetes Deployment YAML defines:
- Number of replicas.
- Image to use (from Docker Hub).
- Port configuration.

**Deployment YAML Example**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-api-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: flask-api
  template:
    metadata:
      labels:
        app: flask-api
    spec:
      containers:
      - name: flask-api
        image: modeha/flask-api:latest
        ports:
        - containerPort: 5000
```

---

#### **Step 4: Exposing the API**
A Kubernetes Service exposes the API internally or externally (e.g., via NodePort).

**Service YAML Example**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: flask-api-service
spec:
  selector:
    app: flask-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: NodePort
```

---

#### **Step 5: Using the Streamlit Interface**
Your Streamlit app (`app.py`) sends requests to the API to predict house prices based on user inputs.

---

### 4. **Running the Pipeline**

1. **Build and Push Docker Image**:
   ```bash
   docker build -t modeha/my-app:latest .
   docker push modeha/my-app:latest
   ```

2. **Run the Pipeline Script**:
   ```bash
   ./run_pipeline.sh my-app
   ```
   This:
   - Kills processes blocking the required port.
   - Trains the model (`train.py`) using a Kubernetes Job.
   - Deploys the API and exposes it.

3. **Access the API via Streamlit**:
   - Launch `app.py`:
     ```bash
     streamlit run app.py
     ```
   - Input house features and get predictions.

---

### 5. **Next Steps**
- **Scaling**: Adjust replicas in your Deployment YAML to scale the API.
- **Monitoring**: Use Kubernetes tools like `kubectl logs`, Prometheus, or Grafana.
- **CI/CD Integration**: Automate deployments with Jenkins, GitHub Actions, or other CI/CD tools.


