# MOE_PADDING=0 python3 -m sglang.launch_server --model /data//lmzheng-grok-1/  --tp 8 --quantization fp8 --trust-remote-code 
python3 -m sglang.launch_server --model-path /data2/lmzheng-grok-1/ --attention-backend wave --tp 8 --quantization fp8 --trust-remote-code 
python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 5

#python3 -m sglang.launch_server --disable-cuda-graph --model /data//lmzheng-grok-1/ --attention-backend wave --tp 8 --quantization fp8 --trust-remote-code 2>&1 | tee "server.log"
