# MPLL
Title: Partial Label Learning via Meta-learning Bi-Level Optimization

```
@INPROCEEDINGS{11308538,
  author={Song, Jiayin and Fu, Yanzhuo and Lu, Wenpeng and Gan, Min and Huang, Linqing and Fan, Jinfu},
  booktitle={2025 International Conference on New Trends in Computational Intelligence (NTCI)}, 
  title={Partial Label Learning via Meta-learning Bi-Level Optimization}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Metalearning;Measurement;Weak supervision;Interference;Feature extraction;Robustness;Data models;Phase locked loops;Optimization;partial label learning;meta-learning;dynamic label correction;bi-level optimization},
  doi={10.1109/NTCI67886.2025.11308538}}
```

## Running MPLL

You need to download MNIST, Kuzushiji-MNIST and Fashion-MNIST datasets into './data/'.

Please note that when partial_rate ∈ {0.1, 0.3, 0.5}.

**Run mnist**

```shell
python -u train.py --ds mnist --partial_rate 0.5
```

**Run kmnist**

```shell
python -u train.py --ds kmnist --partial_rate 0.5 
```

**Run kmnist**

```shell
python -u train.py --ds kmnist --partial_rate 0.5
```
