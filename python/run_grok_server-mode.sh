for reqrate in 8
# for reqrate in 1 2 4 8 16
do
    for iter in $(seq 1 2)
    do
    num_prompts=`echo "300*${reqrate}" | bc -l`
    if [ ${num_prompts} -gt 2400 ];then
         num_prompts=2400
    fi
    num_prompts=3000
    python3 -m sglang.bench_serving --backend sglang --tokenizer Xenova/grok-1-tokenizer --dataset-name random --random-input 1024 --random-output 1024 --num-prompts ${num_prompts} --request-rate $reqrate 2>&1 | tee -a online_bs${num_prompts}i1ko1k_reqrate${reqrate}_tp8_dp1_add-256-512-bs-in-graph_sglang-0.4.1post4_nan-detection_250211.log
    done
done
