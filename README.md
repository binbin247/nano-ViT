# ViT

这是一个使用 PyTorch 实现 Vision Transformer（ViT）的图像分类项目。

## 目录结构

```text
ViT/
├── train.py                         # 训练入口、验证、checkpoint 和训练曲线
├── model.py                         # ViT 模型及各个 Transformer 模块
├── data_loader.py                   # 数据集、数据增强和 DataLoader
├── optim.py                         # AdamW 优化器构建
├── requirements.txt                 # Python 依赖
├── script/
│   └── cifar10_b16_224.sh           # CIFAR10 ViT-B/16 实验脚本
├── data/                            # 数据集目录，不提交到 Git
├── runs/                            # 实验运行结果，不提交到 Git
│   └── cifar10_b16_224/
│       └── <run_id>/
│           ├── args.txt             # 本次运行的参数
│           ├── metrics.json         # loss、accuracy 和 learning rate
│           ├── training_curves.png  # 训练曲线
│           ├── checkpoints/
│           │   ├── last.pt          # 最近一次 checkpoint
│           │   └── best.pt          # 验证集效果最好的 checkpoint
│           └── logs/
│               └── train.log       # 终端训练日志
├── .venv/                           # Python 虚拟环境，不提交到 Git
├── .gitignore
└── README.md
```

## 运行实验

在项目根目录执行：

```bash
./script/cifar10_b16_224.sh
```

脚本会为每次运行创建新的时间戳目录，因此不同实验不会相互覆盖。实验配置直接写在对应的 shell 脚本中。
