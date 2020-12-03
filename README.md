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
--tb_path ./runningSavePath/tbPath \ # tensorboard保存的位置
--log_txt_path ./runningSavePath/logPath \ # 运行时的日志文件保存的位置
--result_path ./runningSavePath/resultPath \ # 训练结束后测试和计算的保存位置
--test_data_folder ./data/campusFrames # 测试数据集的文件夹，即所有视频帧图像所在的文件夹
```

# 数据准备
数据集的安排按照ImageNet的形式，即
```
data_folder
    |class 000000
        |image000000.png
        |image000001.png
        |...
    |class 000001
        |image00000X.png
        |...
    |...
```
**特别注意**，文件夹的名字一定要通过向前补零的方式使它们名字等长！！！照片的名字也需要通过向前补零使它们名字等长！！！这样做可以使在排序时它们的按字典序排序与按数值大小排序的结果保持一致。
如果照片名字没补零的话不影响训练，但是会影响对全部视频帧特征值计算的索引；文件夹名字没补零的话训练都无法进行。