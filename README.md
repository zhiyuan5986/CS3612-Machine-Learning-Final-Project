2023.4.25 下午：Task1取得了阶段性成果，已经基本上完成，接下来先做Task3，如果后面有空再把PCA和tSNE自己实现一遍。

2023.5.20 实现tSNE的道路曲折艰难，目前对于如何求P矩阵，需要进一步探索一下，如果采用https://github.com/pixas/CS3612FinalProject/blob/main/scripts/utils.py的方法，应该可以找到比较好的sigma，但是如果采用https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/，求解过程中似乎会出现溢出的情况。

最新更新：第一种方法可以比较好地求出P，但是似乎优化起来没有什么改进。