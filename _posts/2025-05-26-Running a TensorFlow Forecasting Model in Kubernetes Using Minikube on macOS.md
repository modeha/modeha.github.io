# 📘 Running a TensorFlow Forecasting Model in Kubernetes Using Minikube on macOS

This guide explains step-by-step how to set up, deploy, and run a time-series forecasting model in a local Kubernetes environment using Minikube, Docker, and TensorFlow on macOS. The process includes building a Docker container, defining Kubernetes resources, and executing predictions.

---

## 📁 Project Structure

Here's the directory layout:

```bash
ci_cp/
├── Dockerfile
├── requirements.txt
├── load_saved_model.py
├── mlp.keras
├── test_data.csv
├── deployment.yaml
└── run_forecast_k8s.sh
```

Each file plays a role:

* `Dockerfile`: builds the container image.
* `requirements.txt`: defines Python dependencies.
* `load_saved_model.py`: loads the model and makes predictions.
* `mlp.keras`: pre-trained model file.
* `test_data.csv`: test input data.
* `deployment.yaml`: Kubernetes deployment definition.
* `run_forecast_k8s.sh`: optional automation script.

---

## 🔧 Step 1 – Install Required Tools

Install the following tools via Homebrew:

```bash
brew install --cask docker
brew install minikube
brew install kubectl
```

* **Docker Desktop**: Runs containers and includes Kubernetes support.
* **Minikube**: Creates a local Kubernetes cluster.
* **kubectl**: CLI tool to interact with Kubernetes.

Start Docker Desktop and enable Kubernetes in its settings.

---

## 🔁 Step 2 – Start Minikube

Start your local Kubernetes cluster:

```bash
minikube start --driver=docker
```

Check the node status:

```bash
kubectl get nodes
```

You should see a `Ready` node.

---

## 📂 Step 3 – Mount Local Directory into Minikube

Minikube runs in a VM/container, so we must mount the local project path into it:

```bash
minikube mount "/Users/mohsendehghani/Desktop/Projects/ci_cp:/data/well-forecasting"
```

> ⚠️ Keep this terminal open as it holds the mount session.

This makes your macOS directory accessible inside Kubernetes pods at `/data/well-forecasting`.

---

## 🐳 Step 4 – Build the Docker Image Inside Minikube

To ensure the image is visible to Kubernetes:

```bash
eval $(minikube docker-env)
docker build -t well-forecasting:latest .
```

This switches your terminal to use Minikube's internal Docker daemon and builds the image `well-forecasting:latest`.

---

## 📦 Step 5 – Kubernetes Deployment YAML Breakdown

Here is the full `deployment.yaml` used to deploy the forecasting container:

```yaml
apiVersion: apps/v1               # API version for Deployment object
kind: Deployment                  # Specifies this is a Deployment
metadata:
  name: mlp-predictor             # Deployment name
spec:
  replicas: 1                     # Number of pods
  selector:
    matchLabels:
      app: mlp-predictor         # Label selector for pods
  template:
    metadata:
      labels:
        app: mlp-predictor       # Label applied to pod
    spec:
      containers:
      - name: mlp-container      # Name of the container
        image: well-forecasting:latest      # Built Docker image
        imagePullPolicy: Never             # Don't pull from Docker Hub
        workingDir: /app                   # Working directory in container
        stdin: true
        tty: true
        volumeMounts:
        - mountPath: /app/data             # Inside-container path
          name: data-volume                # Must match volume below
      volumes:
      - name: data-volume
        hostPath:
          path: /data/well-forecasting     # Minikube-mount path
```

### 🔍 Explanation

* `volumeMounts` mounts the `/data/well-forecasting` folder into `/app/data` inside the container.
* `imagePullPolicy: Never` avoids pulling from Docker Hub.
* `replicas: 1` means only one pod will run.

Apply the deployment:

```bash
kubectl apply -f deployment.yaml
```

---

## 📊 Step 6 – Run the Model in the Pod

Find the pod:

```bash
kubectl get pods
```

Then run your model inside the container:

```bash
kubectl exec -it <your-pod-name> -- python load_saved_model.py
```

When prompted:

```
Enter model name to load (e.g., MLP): mlp
Enter number of input time steps used in training: 168
```

Expected output:

```
✅ Loaded model: mlp
✅ Prediction complete
📁 Predictions saved to data/predictions.csv
```

---

## 📤 Step 7 – Copy Predictions to Host

To retrieve the prediction output:

```bash
kubectl cp <your-pod-name>:/app/data/predictions.csv ./predictions.csv
```

This downloads the file to your macOS host.

---

## 🧠 Summary: How It All Works

1. Your macOS folder is mounted into the Minikube VM.
2. Docker builds an image that contains TensorFlow and your script.
3. Kubernetes runs a pod using that image and accesses data via the mount.
4. The script loads the `.keras` model and test data.
5. It outputs a CSV with predictions, which you retrieve with `kubectl cp`.

---

## 🔄 Optional: Shell Script for Automation

To automate the process, use a shell script `run_forecast_k8s.sh`:

```bash
chmod +x run_forecast_k8s.sh
./run_forecast_k8s.sh
```

This can start Minikube, build the image, apply the deployment, and run your model—all in one go.

---

This setup lets you simulate production pipelines locally using Kubernetes and is a powerful way to validate ML workflows in containers.
