# ai-inference-benchmark
PyTorch image classification inference + latency benchmarking
# AI Image Inference & Benchmarking (PyTorch)

This project implements an image classification pipeline using a pretrained ResNet18 model in PyTorch. It performs batch inference on local images and measures latency.

## Features
- Uses pretrained ResNet18 for image classification
- Supports CPU and Apple GPU (MPS)
- Batch processing for improved performance
- Measures average inference latency
- Outputs top-5 predictions with confidence scores
- Exports results to CSV

## Tech Stack
- Python
- PyTorch
- torchvision

## How to Run

1. Install dependencies:
pip install torch torchvision pillow

2. Add your images to a folder (update path in code)

3. Run:
python inference_benchmark.py

## Example Output
- Average latency (ms)
- Top-5 predictions per image
- predictions.csv file with results

## Notes
- Automatically falls back to CPU if GPU (MPS) is unavailable

## Results

- Average inference latency: XX ms
- Batch size: 8
- Device: CPU / Apple MPS

Example output:
Image: dog.jpg  
1. Labrador Retriever: 92.3%  
2. Golden Retriever: 5.1%
