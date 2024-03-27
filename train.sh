export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1  #2,3
export NCCL_P2P_DISABLE="1" #NCCL(p2p) 통신 비활성화
export NCCL_IB_DISABLE="1" #NCLL InfiniBand 통신 비활성화

master_port=$(shuf -i 29500-29999 -n 1)

echo "Selected master port: $master_port\n"

torchrun \
  --nproc_per_node 1 \
  --master_port "$master_port" \
  main.py