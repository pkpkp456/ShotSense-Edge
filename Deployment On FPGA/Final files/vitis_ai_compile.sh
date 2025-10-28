#!/bin/bash
ARCH_PATH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json"
MODEL_INT8="./quantized_model/deploy_model.onnx"
OUT_DIR="./compiled_model"

echo "⚙️ Compiling for FPGA (ZCU104)..."
vai_c_xir   --xmodel ${MODEL_INT8}   --arch ${ARCH_PATH}   --output_dir ${OUT_DIR}   --net_name gunshot_embedded

echo "✅ FPGA model generated: ${OUT_DIR}/gunshot_embedded.xmodel"
