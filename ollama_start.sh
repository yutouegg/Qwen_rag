#!/bin/bash

echo "Starting Ollama service..."
ollama serve &

sleep 5

# 启动qwen
echo "Running qwen2.57b model..."
ollama run qwen2.5:7b &

