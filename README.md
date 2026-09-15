# ODAS Cao 数据集近似复现

本项目依据论文 **An Open-World Semi-Supervised Recognition Method (ODAS) for Φ-OTDR Disturbance Signals** 的公开信息，实现 Cao 实验室数据上的可运行近似复现。Wu 现场数据、Wu 实验和环境配置暂不包含在本阶段。

## 已实现内容

- Cao `.mat` 数据读取；
- 论文规定的 `10000×12 → 32×32` 预处理：时间平均池化、空间插值、归一化；
- ResNet-18 单通道 backbone、分类头和投影头；
- 监督学习基线；
- ODAS 近似复现的 UCM、memory smoothing、伪标签图对比学习和 CACM；
- 开放世界已知类准确率、未知类 Hungarian 聚类准确率、总体准确率、宏平均指标和混淆矩阵；
- 每轮 JSON 日志、模型权重和配置保存。

论文没有公开官方 ODAS 源码，也没有明确给出全部分类头、投影头、增强操作和部分训练细节。本项目中这些内容均属于 `reproduction assumption`，可在 JSON 配置中修改，不能视为作者原始实现。

## 数据目录

将 Cao 数据放在 `Data/train` 和 `Data/test`，每个 split 下保留六个类别文件夹及 `label.txt`。数据来源和下载地址见 `Data/readme.txt`。原始 `.mat` 和压缩包不会提交到 GitHub。

## 服务器运行

在服务器项目根目录执行：

```bash
python -m pip install -r requirements.txt
python train.py --config configs/cao_supervised.json --mode supervised
python train.py --config configs/cao_odas.json --mode odas
python evaluate.py --checkpoint runs/cao_odas/last.pt --config configs/cao_odas.json
```

建议先将 `epochs` 改为 1，确认数据、模型和指标流程都能运行，再开始正式实验。

## 输出

每次训练保存到配置文件的 `output_dir`，包括 `last.pt` 和 `history.json`。训练中不读取未标注样本真实标签，测试阶段才使用标签评价。

## 复现边界

当前实现目标是论文公开信息下的工程近似复现。论文中的完整对比方法 ORCA、NACH、OpenNCD、SSOC、OpenCon、TRSSL，以及 Wu 现场数据实验尚未实现。论文报告的数值只能作为参考目标，最终结果需要依据服务器环境、随机种子和数据划分单独记录。
