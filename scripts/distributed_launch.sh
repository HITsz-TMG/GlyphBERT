PORT=5622
NPROC=4
python -m torch.distributed.launch --nproc_per_node=${NPROC} --master_port=${PORT} distributed_main.py --parallel=DDP

# nproc_per_node 参数指定为当前主机创建的进程数。一般设定为当前主机的 GPU 数量。比如单机8卡就是8
# master_port为master节点的端口号
