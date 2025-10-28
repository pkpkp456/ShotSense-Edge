#!/bin/bash
MODEL_FP32="gunshot_embedded_fp32.onnx"
OUTPUT_DIR="quantized_model"

echo "🔧 Running Vitis AI quantizer..."
vai_q_onnx quantize   --model ${MODEL_FP32}   --output_dir ${OUTPUT_DIR}   --calib_dataset ./   --calib_iter 100   --input_nodes input   --output_nodes output

echo "✅ Quantization complete. Output: ${OUTPUT_DIR}/deploy_model.onnx"
