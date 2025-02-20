#!/bin/bash
 
# export SGLANG_TORCH_PROFILER_DIR=/data/sglang/
export SGLANG_TORCH_PROFILER_DIR=/home/amd/jacky/jacky/sglang/python
 
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
 
# Define the log file with a timestamp
LOGFILE="sglang_server_log_$TIMESTAMP.json"

# Run the Python command and save the output to the log file
RCCL_MSCCL_ENABLE=0 DEBUG_HIP_BLOCK_SYNC=1024 GPU_FORCE_BLIT_COPY_SIZE=64 python3 -m sglang.launch_server \
    --model-path  /data2/lmzheng-grok-1/ \
    --tokenizer-path Xenova/grok-1-tokenizer \
    --tp 8 --quantization fp8 --trust-remote-code --disable-radix-cache --cuda-graph-max-bs 512 \
    --port 30001 2>&1 | tee "$LOGFILE"