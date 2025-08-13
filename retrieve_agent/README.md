### running shell
``` bash 
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-4B --dtype auto  --port 8008 --gpu-memory-utilization 0.3
python colqwen.py 
bash weaviate-start.sh
uvicorn app.main:app --port 8023 --reload
185d-44ab-b145-28d69150ff46
```