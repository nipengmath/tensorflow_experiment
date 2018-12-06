# 异步
## 登录机器17，启动ps:0
CUDA_VISIBLE_DEVICES=-1 python dist_tf_mnist.py --dist async --ps_hosts xx.xx.xx.17:2223 --worker_hosts xx.xx.xx.17:2224,xx.xx.xx.17:2225,xx.xx.xx.18:2223,xx.xx.xx.18:2224 --job_name ps --task_index 0
## 登录机器17，启动worker:0
CUDA_VISIBLE_DEVICES=0 python dist_tf_mnist.py --dist async --ps_hosts xx.xx.xx.17:2223 --worker_hosts xx.xx.xx.17:2224,xx.xx.xx.17:2225,xx.xx.xx.18:2223,xx.xx.xx.18:2224 --job_name worker --task_index 0
## 登录机器17，启动worker:1
CUDA_VISIBLE_DEVICES=1 python dist_tf_mnist.py --dist async --ps_hosts xx.xx.xx.17:2223 --worker_hosts xx.xx.xx.17:2224,xx.xx.xx.17:2225,xx.xx.xx.18:2223,xx.xx.xx.18:2224 --job_name worker --task_index 1
## 登录机器18，启动worker:2
CUDA_VISIBLE_DEVICES=0 python dist_tf_mnist.py --dist async --ps_hosts xx.xx.xx.17:2223 --worker_hosts xx.xx.xx.17:2224,xx.xx.xx.17:2225,xx.xx.xx.18:2223,xx.xx.xx.18:2224 --job_name worker --task_index 2
## 登录机器18，启动worker:3
CUDA_VISIBLE_DEVICES=1 python dist_tf_mnist.py --dist async --ps_hosts xx.xx.xx.17:2223 --worker_hosts xx.xx.xx.17:2224,xx.xx.xx.17:2225,xx.xx.xx.18:2223,xx.xx.xx.18:2224 --job_name worker --task_index 3


# 同步
## 登录机器17，启动ps:0
CUDA_VISIBLE_DEVICES=-1 python dist_tf_mnist.py --dist sync --ps_hosts xx.xx.xx.17:2223 --worker_hosts xx.xx.xx.17:2224,xx.xx.xx.17:2225,xx.xx.xx.18:2223,xx.xx.xx.18:2224 --job_name ps --task_index 0
## 登录机器17，启动worker:0
CUDA_VISIBLE_DEVICES=0 python dist_tf_mnist.py --dist sync --ps_hosts xx.xx.xx.17:2223 --worker_hosts xx.xx.xx.17:2224,xx.xx.xx.17:2225,xx.xx.xx.18:2223,xx.xx.xx.18:2224 --job_name worker --task_index 0
## 登录机器17，启动worker:1
CUDA_VISIBLE_DEVICES=1 python dist_tf_mnist.py --dist sync --ps_hosts xx.xx.xx.17:2223 --worker_hosts xx.xx.xx.17:2224,xx.xx.xx.17:2225,xx.xx.xx.18:2223,xx.xx.xx.18:2224 --job_name worker --task_index 1
## 登录机器18，启动worker:2
CUDA_VISIBLE_DEVICES=0 python dist_tf_mnist.py --dist sync --ps_hosts xx.xx.xx.17:2223 --worker_hosts xx.xx.xx.17:2224,xx.xx.xx.17:2225,xx.xx.xx.18:2223,xx.xx.xx.18:2224 --job_name worker --task_index 2
## 登录机器18，启动worker:3
CUDA_VISIBLE_DEVICES=1 python dist_tf_mnist.py --dist sync --ps_hosts xx.xx.xx.17:2223 --worker_hosts xx.xx.xx.17:2224,xx.xx.xx.17:2225,xx.xx.xx.18:2223,xx.xx.xx.18:2224 --job_name worker --task_index 3
