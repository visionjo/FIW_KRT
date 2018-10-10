# SphereFace
An RFIW Kinship Verification Baseline using SphereFace.

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

Code heavily borrowed from [sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch)

Includes the following pretrained model (also from above repo): [20171020](model/sphere20a_20171020.7z)

# Train + Validation
```
python main.py
```

# Finetune + Validation
```
python main.py --finetune
```

# Validation (with the model: sphere20a_2.pth)
```
python main.py --no_train --pretrained sphere20a_2.pth
```
