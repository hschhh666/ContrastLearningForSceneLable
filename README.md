# 使用示例

运行以下脚本即可开始训练、测试正负样本平均距离和计算所有图像的特征值
```python
CUDA_VISIBLE_DEVICES=0 \ # 指定GPU
python train_CLSL.py \ # 主程序
--batch_size 8 \
--num_workers 36 \
--contrastMethod e2e \ # 训练方法，基于端到端或memorybank
--nce_k 10 \ # 负样本个数
--print_freq 10 \ # 输出信息的频率
--epochs 40 \ # 训练轮数
--save_freq 10 \ # 每多少个epoch保存一次
--feat_dim 128 \ # 网络输出的特征维度
--data_folder ./data/campusSceneDatasetForTest \ # 训练数据集
--model_path ./runningSavePath/modelPath \ # 模型保存的位置
--tb_path ./runningSavePath/tbPath \ tensorboard保存的位置
--log_txt_path ./runningSavePath/logPath \ # 运行时的日志文件保存的位置
--result_path ./runningSavePath/resultPath \ # 训练结束后测试和计算的保存位置
--test_data_folder ./data/campusFrames # 测试数据集的文件夹，即所有视频帧图像所在的文件夹
```

# 数据准备
数据集的安排按照ImageNet的形式，即
```
data_folder
    |class 0
        |image0.png
        |image1.png
        |...
    |class 1
        |imageX.png
        |...
    |...
```