# LLM-Powered Network Optimization Advisor - Docker Setup

This repository contains the Dockerized version of the LLM-Powered Network Optimization Advisor based on a fine-tuned Llama-3-8B model.

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/meddhiabetis/LLM-Powered-Network-Optimization-Advisor.git
   cd LLM-Powered-Network-Optimization-Advisor
   ```

2. **Place your model files** inside the `app/network_optimizer/` directory.  
   > ⚠️ **Note:** Do not commit large model files to the repository! They are ignored by `.gitignore`.

3. **Build and run the Docker container:**
   ```bash
   docker-compose up --build
   ```

## API Usage

The API will be available at `http://localhost:8000`

### Endpoints

1. **POST /optimize**
   - Endpoint for network optimization.
   - Example request:
     ```bash
     curl -X POST "http://localhost:8000/optimize" \
          -H "Content-Type: application/json" \
          -d '{"metrics": {"metric1": "value1", "metric2": "value2"}}'
     ```

2. **GET /health**
   - Health check endpoint.
   - Returns the status of the service.

## Environment Variables

- `MODEL_PATH`: Path to the model files inside the container.
- `CUDA_VISIBLE_DEVICES`: GPU device ID to use.

## Monitoring

The API includes a basic health check at the `/health` endpoint.  
For production deployment, consider adding:
- Prometheus metrics
- Grafana dashboards
- Log aggregation
