# Conda/venv commands
conda activate <env_name>

# Replace LD_LIBRARY_PATH with the path to the nvjitlink library in the conda environment (not sure if this is needed/not)
export LD_LIBRARY_PATH=<env_base_path>/lib/<python_version - python3.11>/site-packages/nvidia/nvjitlink/lib/:$LD_LIBRARY_PATH

# HF model name, appropriate tensor parallel size, ideally check all the parameters
CUDA_VISIBLE_DEVICES=0,1<2,3> vllm serve <HF_model_name/path on the scratch> --tensor-parallel-size <2/4> --gpu-memory-utilization 0.85 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port 8080 &

# We want to shut down the VLLM server after the experiment is done, so we need its PID
VLLMPID=$!

# Run the experiment, possible to run multiple experiments in sequence
python flipflop/prompt_flipflop.py --ip_path datasets/flipflop/examples.txt --save_path results/flipflop/OLMo_7B/ --engine openai

# Shut down the VLLM server
kill $VLLMPID
