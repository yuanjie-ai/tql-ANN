#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : tql-ANN.
# @File         : ANN
# @Time         : 2019-12-04 20:05
# @Author       : yuanjie
# @Email        : yuanjie@xiaomi.com
# @Software     : PyCharm
# @Description  :


import faiss
import numpy as np


class ANN(object):

    def __init__(self):
        self.nogpu_index_factory = {'HNSW', 'SQ'}

    def train(self, data, index_factory='Flat', metric=None, noramlize=False):
        """

        :param data:
        :param index_factory:
            https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
            https://github.com/liqima/faiss_note/blob/master/4.Faiss%20indexes%20IO%E5%92%8Cindex%20factory.ipynb
        :param metric:
            faiss.METRIC_BrayCurtis
            faiss.METRIC_Canberra
            faiss.METRIC_INNER_PRODUCT: 对于归一化之后的向量等价于 cosine
            faiss.METRIC_JensenShannon
            faiss.METRIC_L1
            faiss.METRIC_L2
            faiss.METRIC_Linf
            faiss.METRIC_Lp
        :return:
        """
        if noramlize:
            data = self.noramlize(data)
        assert data.dtype == 'float32', "TODO: np.array([]).astype('float32')"
        dim = data.shape[1]
        args = [dim, index_factory, metric] if metric else [dim, index_factory]

        self.index = faiss.index_factory(*args)

        if faiss.get_num_gpus() > 0:
            if any(index_factory.__contains__(i) for i in self.nogpu_index_factory):
                pass
                print(f"Donot Support GPU: {index_factory}")
            else:
                # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
                self.index = faiss.index_cpu_to_all_gpus(self.index)

        print(f"Train ...")
        self.index.train(data)
        self.index.add(data)

        print(f"索引总量: {self.index.ntotal}")

    def search(self, data, topK=10, nprobe=1, k_factor=1):
        """

        :param data:
        :param topK:
        :param nprobe: nprobe参数始终是调整速度和结果精度之间权衡的一种方式。
        :param k_factor:
        :return:
        """
        self.index.k_factor = k_factor  # 搜索阶段会首先选取 k_factor*topK，重排序
        self.index.nprobe = nprobe  # default nprobe is 1, try a few more
        return self.index.search(data, topK)

    def write_index(self, file_name="index_file.index"):
        faiss.write_index(self.index, file_name)

    def read_index(self, file_name="index_file.index"):
        index = faiss.read_index(file_name)
        # if faiss.get_num_gpus() > 0:
        #     index = faiss.index_cpu_to_gpu()
        # index_new = faiss.clone_index(index) # 复制索引
        return index

    def noramlize(self, x):
        if len(x.shape) > 1:
            return x / np.clip(x ** 2, 1e-12, None).sum(axis=1).reshape((-1, 1) + x.shape[2:]) ** 0.5
        else:
            return x / np.clip(x ** 2, 1e-12, None).sum() ** 0.5
