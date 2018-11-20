Hierarchical Attention Networks for Document Classification. TensorFlow 版本的实现, 完全按照paper中介绍的网络结构

1. 准备原始训练数据
python data_process.py
2. 整理训练数据, 保存为tfrecords格式
./prepare.sh
3. 训练模型
./train.sh
4. 预测
python batch_inference.py
