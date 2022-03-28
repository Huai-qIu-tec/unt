import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


class DriveDataSet(Dataset):
    def __init__(self, root, train=True, transforms=None):
        super(DriveDataSet, self).__init__()
        if train:
            self.flag = "training"
        else:
            self.flag = "test"
        data_root = os.path.join(root, "../UNet_project/DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists"
        self.transforms = transforms
        # img的名字
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        # img的路径
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # img的label的路径
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # 确认label文件是否存在
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists")

        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # 确认掩膜文件是否存在
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 0-有用信息 255-无用信息
        manual = Image.open(self.manual[idx]).convert('L')
        # 1-前景 0-背景
        manual = np.array(manual) / 255
        # 255-有用信息 0-无用信息
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # 0-有用信息 255-无用信息
        roi_mask = 255 - np.array(roi_mask)
        # 1-有用+前景 0-背景 255-无用, 且把小于0的像素和大于255的像素值截断
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 转PIL transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        '''
        :param batch: 形式为bathch_size * [source_img, target_img],即每个batch中
                      如batch[0, 0] = source_img.shape = [3, 480, 480], batch[0, 1] = target_img.shape = [480, 480]
        zip(*batch): 把batch解开之后按列表的0和1进行打包, 结果变成source_img[shape]和target_img[shape]
        :return:
        '''
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch中channel, H, W的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # batched_shape = [batch_size, channels, h, w]
    batched_shape = (len(images),) + max_size
    # 构建新的batch_img, 用fill_value填充
    batched_imgs = images[0].new(*batched_shape).fill_(fill_value)
    # 把img里的照片copy到pad_img
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)

    return batched_imgs
