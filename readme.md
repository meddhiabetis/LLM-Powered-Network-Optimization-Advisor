# Network Optimizer Docker Setup

This repository contains the Dockerized version of the Network Optimization AI Assistant based on fine-tuned Llama-3-8B model.

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Setup

1. Clone this repository:
```bash
git clone https://github.com/meddhiabetis/network-optimizer-docker.git
cd network-optimizer-docker
```

2. Place your `network_optimizer.zip` file in the `model/` directory.

3. Build and run the Docker container:
```bash
docker-compose up --build
```

## API Usage

The API will be available at `http://localhost:8000`

### Endpoints:

1. **POST /optimize**
   - Endpoint for network optimization
   - Example request:
   ```bash
   curl -X POST "http://localhost:8000/optimize" \
        -H "Content-Type: application/json" \
        -d '{"metrics": {"metric1": "value1", "metric2": "value2"}}'
   ```

2. **GET /health**
   - Health check endpoint
   - Returns status of the service

## Environment Variables

- `MODEL_PATH`: Path to the model files inside container
- `CUDA_VISIBLE_DEVICES`: GPU device ID to use

## Monitoring

The API includes basic health checking at the `/health` endpoint. For production deployment, consider adding:
- Prometheus metrics
- Grafana dashboards
- Log aggregation

## License

[Your License Here]