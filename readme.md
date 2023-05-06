- Docker build:
    docker build -t gradio_app .
- Docker run:
    docker run -d -p 8000:8000 -e GRADIO_SERVER_NAME=0.0.0.0 gradio_app