# -*- coding: utf-8 -*-

from keras import backend as K
import torch
from torch import Tensor

class sim_dis():


    def jaccard_loss(
            input1: Tensor,
            input2: Tensor,
            smooth: float = 1e-10,
    ) -> Tensor:
        r"""jaccard_similarity(input1, input2, smooth=1e-10) -> Tensor

        计算两个张量的杰卡德相似度。

        Args:
            input1 (Tensor): 第一个输入张量。
            input2 (Tensor): 第二个输入张量。
            smooth (float, optional): 平滑项，避免除零错误。默认值是 1e-10。

        Returns:
            Tensor: 杰卡德相似度。

        Example:
            >>> input1 = torch.tensor([1, 2, 3])
            >>> input2 = torch.tensor([2, 3, 4])
            >>> jaccard_similarity(input1, input2)
            tensor(0.5000)
        """
        result=[]
        for i in range(len(input1)):
            set1 = input1[i]
            set2 = input2[i]
            # intersection = K.sum(K.abs(y_true * y_pred), axis=axis_reduce)
            intersection = torch.sum(set1 * set2)
            #  sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=axis_reduce)
            union = torch.sum(set1) + torch.sum(set2) - intersection
            # jaccard = (intersection + smooth) / (sum_ - intersection + smooth)
            jaccard = (intersection + smooth) / (union + smooth)
            result.append(jaccard)

        return torch.Tensor(result).to('cuda:0')

    def jaccard_loss2(
            input1: Tensor,
            input2: Tensor,
            smooth: float = 1e-10,
    ) -> Tensor:
        r"""jaccard_similarity(input1, input2, smooth=1e-10) -> Tensor

        计算两个张量的杰卡德相似度。

        Args:
            input1 (Tensor): 第一个输入张量。
            input2 (Tensor): 第二个输入张量。
            smooth (float, optional): 平滑项，避免除零错误。默认值是 1e-10。

        Returns:
            Tensor: 杰卡德相似度。

        Example:
            >>> input1 = torch.tensor([1, 2, 3])
            >>> input2 = torch.tensor([2, 3, 4])
            >>> jaccard_similarity(input1, input2)
            tensor(0.5000)
        """
        # intersection = K.sum(K.abs(y_true * y_pred), axis=axis_reduce)
        intersection = torch.sum(input1 * input2)
        #  sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=axis_reduce)
        union = torch.sum(input1) + torch.sum(input2) - intersection
        # jaccard = (intersection + smooth) / (sum_ - intersection + smooth)
        jaccard = (intersection + smooth) / (union + smooth)


        return jaccard
