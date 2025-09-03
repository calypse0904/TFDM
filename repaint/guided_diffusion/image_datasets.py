import random
import os
from os.path import expanduser

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import imwrite
from XImage import CXImage


def to_file_ext(img_names, ext):
    img_names_out = []
    for img_name in img_names:
        splits = img_name.split('.')
        if not len(splits) == 2:
            raise RuntimeError("File name needs exactly one '.':", img_name)
        img_names_out.append(splits[0] + '.' + ext)

    return img_names_out


def log_img_path(txt_path, img_path):
    with open(txt_path, "a+") as file:
        file.seek(0, 2)
        if file.tell() != 0:  # 如果文件不为空，则在写入新行之前换行
            file.write("\n")
        file.write(img_path)


def write_images(imgs, img_names, dir_path):  # 保存图像
    os.makedirs(dir_path, exist_ok=True)

    for image_name, image in zip(img_names, imgs):
        out_path = os.path.join(dir_path, image_name)
        imwrite(img=image, path=out_path)


def eval_imswrite(srs=None, img_names=None, dset=None, name=None, ext='tif', lrs=None, gts=None, gt_keep_masks=None,
                  conf=None, verify_same=True):
    img_names = to_file_ext(img_names, ext)  # 设置图像名称

    if srs is not None:
        sr_dir_path = expanduser(conf.data.eval.paths.srs)
        write_images(srs, img_names, sr_dir_path)

    if gt_keep_masks is not None:
        mask_dir_path = expanduser(
            conf.data.eval.paths.gt_keep_masks)
        write_images(gt_keep_masks, img_names, mask_dir_path)

    gts_path = conf.data.eval.paths.gts
    if gts is not None and gts_path:
        gt_dir_path = expanduser(gts_path)
        write_images(gts, img_names, gt_dir_path)

    if lrs is not None:
        lrs_dir_path = expanduser(
            conf.data.eval.paths.lrs)
        write_images(lrs, img_names, lrs_dir_path)


def load_data_inpa(
        conf,
        model_flag='train',
        **kwargs
):
    if model_flag == 'eval':
        gt_dir = os.path.expanduser(conf.data.eval.gt_path)  # 将路径转换成绝对路径
        gt_paths = _list_image_files_recursively(gt_dir)  # 获取符合指定格式的图像路径列表

        mask_paths = ''
        refer_paths = ''
        mask_dir = os.path.expanduser(conf.data.eval.mask_path)
        mask_paths = _list_image_files_recursively(mask_dir)
        if conf.condition.condition_flag:
            refer_dir = os.path.expanduser(conf.data.eval.refer_path)
            refer_paths = _list_image_files_recursively(refer_dir)

        assert len(gt_paths) == len(mask_paths)
        random_crop = conf.data.eval.random_crop
        random_flip = conf.data.eval.random_flip
        return_dict = conf.data.eval.return_dict
        max_len = conf.data.eval.max_len
        offset = conf.data.eval.offset
        mask_loader_flag = True
    else:
        gt_dir = os.path.expanduser(conf.data.train.gt_path)  # 将路径转换成绝对路径
        gt_paths = _list_image_files_recursively(gt_dir)  # 获取符合指定格式的图像路径列表

        mask_paths = ''
        refer_paths = ''
        if conf.condition.condition_flag:
            refer_dir = os.path.expanduser(conf.data.train.refer_path)
            refer_paths = _list_image_files_recursively(refer_dir)
        assert len(gt_paths) == len(refer_paths)
        random_crop = conf.data.train.random_crop
        random_flip = conf.data.train.random_flip
        return_dict = conf.data.train.return_dict
        max_len = conf.data.train.max_len
        offset = conf.data.train.offset
        mask_loader_flag = False

    dataset = ImageDatasetInpa(
        conf.model.image_size,
        gt_paths=gt_paths,
        mask_paths=mask_paths,
        refer_paths=refer_paths,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
        return_dict=return_dict,
        max_len=max_len,
        offset=offset,
        mask_loader_flag=mask_loader_flag,
        condition_flag=conf.condition.condition_flag,
        augment=conf.data.train.augment
    )

    if model_flag == 'eval':  # 是否顺序加载图像
        loader = DataLoader(
            dataset, batch_size=conf.data.eval.batch_size, shuffle=False, num_workers=0,
            drop_last=conf.data.eval.drop_last
        )

    else:
        loader = DataLoader(
            dataset, batch_size=conf.data.train.batch_size, shuffle=True, num_workers=0,
            drop_last=conf.data.train.drop_last
        )

    if model_flag == 'train':
        def generator_func():
            while True:
                for data in loader:
                    yield data

        return generator_func()
    else:
        return loader


def _list_image_files_recursively(data_dir):  # 返回所有符合指定格式的文件列表
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "tif"]:  # 加上tif格式
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDatasetInpa(Dataset):
    def __init__(
            self,
            resolution,
            gt_paths,
            mask_paths,
            refer_paths,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
            return_dict=False,
            max_len=None,
            offset=0,
            mask_loader_flag=True,
            condition_flag="sr",
            augment=False
    ):
        super().__init__()
        self.resolution = resolution

        gt_paths = sorted(gt_paths)[offset:]
        self.local_gts = gt_paths[shard:][::num_shards]
        if mask_loader_flag:
            mask_paths = sorted(mask_paths)[offset:]
            self.local_masks = mask_paths[shard:][::num_shards]
        if condition_flag != 'uncondition':
            refer_paths = sorted(refer_paths)[offset:]
            self.refer_gts = refer_paths[shard:][::num_shards]

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_dict = return_dict
        self.max_len = max_len
        self.mask_loader_flag = mask_loader_flag
        self.condition_flag = condition_flag
        self.augment = augment

    def __len__(self):
        if self.max_len is not None:
            return self.max_len

        return len(self.local_gts)

    def __getitem__(self, idx):
        gt_path = self.local_gts[idx]
        gt_data = self.imread(gt_path)
        if self.condition_flag != 'uncondition':
            refer_path = self.refer_gts[idx]
            refer_data = self.imread(refer_path)
        if self.mask_loader_flag:
            mask_path = self.local_masks[idx]
            mask_data = self.imread(mask_path)

        # txt_path=r'/home/yingying/project/conditional_diffusion/img_record.txt'
        # log_img_path(txt_path,refer_path)

        if self.random_crop:
            raise NotImplementedError()
        else:
            if self.augment == True:
                augment_flag = np.random.randint(0, 3)
            else:
                augment_flag = 0
            arr_gt = center_crop_arr(gt_data, self.resolution, num=augment_flag)
            if self.condition_flag != 'uncondition':
                arr_ref = center_crop_arr(refer_data, self.resolution, num=augment_flag)
            if self.mask_loader_flag:
                arr_mask = center_crop_arr(mask_data, self.resolution, num=augment_flag)  # 返回的图像是np.array形式

        if self.random_flip and random.random() < 0.5:
            arr_gt = arr_gt[:, ::-1]
            arr_ref = arr_ref[:, ::-1]
            if self.mask_loader_flag:
                arr_mask = arr_mask[:, ::-1]

        # arr_gt = arr_gt.astype(np.float32) / 127.5 - 1#将像素值映射到[-1,1]，应该可以自己替换
        # 将dem像素值缩放,用8000作为固定比例
        arr_gt, gt_min, gt_max = dem_zip(arr_gt)
        if self.mask_loader_flag:
            if np.max(arr_mask) > 1:
                arr_mask = arr_mask.astype(np.float32) / 255.0  # 将像素值映射到[0,1]
            arr_mask = np.expand_dims(arr_mask, axis=2)
            arr_mask = np.ascontiguousarray(arr_mask)

        if self.condition_flag == 'sr':  # 超像素情况下进行像素归一化
            # 需要取单通道数据实验
            #arr_ref=arr_ref[:,:,2]   # 👈 0 1 2 分别是 R G B
            arr_ref, ref_min, ref_max = dem_zip(arr_ref)
            arr_ref = np.ascontiguousarray(arr_ref)
        if self.condition_flag == 'tf':
            arr_ref = np.where(arr_ref == 255, -1, arr_ref)
            arr_ref = np.expand_dims(arr_ref, axis=2)
            arr_ref = np.ascontiguousarray(arr_ref)

        if self.return_dict:
            name = os.path.basename(gt_path)
            outdict = {
                'GT': np.transpose(arr_gt, [2, 0, 1]),
                'GT_name': name,
                'gt_min': gt_min,
                'gt_max': gt_max
            }
            if self.mask_loader_flag:
                outdict.update({
                    'gt_keep_mask': np.transpose(arr_mask, [2, 0, 1])
                })
            if self.condition_flag != "uncondition":
                outdict.update({
                    'refer': np.transpose(arr_ref, [2, 0, 1])
                })
            return outdict
        else:
            raise NotImplementedError()

    def imread(self, path):
        img = CXImage()
        img.Open(path)
        data = img.GetData(np.float32)

        return data


def center_crop_arr(arr, image_size, num=0):
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    res = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

    if num == 1:  # 左右翻转
        res = np.fliplr(res)
    elif num == 2:  # 上下翻转
        res = np.flipud(res)
    elif num == 3:  # 顺时针旋转180度
        res = np.rot90(res, 2)
    return res


# def dem_zip(image_array):  # 将dem像素值压缩
#     # 将哨兵2中的nan值替换成0
#
#     if len(image_array.shape) == 2:  # 如果输入的是单通道数据
#
#         # 无数据值不参与归一化
#         valid_mask = image_array != -32767
#         valid_data = image_array[valid_mask]
#         min_value = np.nanmin(valid_data)
#         max_value = np.nanmax(valid_data)
#
#         # min_value = np.nanmin(image_array)
#         # max_value = np.nanmax(image_array)
#
#         data_range = max_value - min_value
#         low_value = min_value - data_range * 0.1
#         high_value = max_value + data_range * 0.1
#
#         image_array = ((image_array - low_value) / (high_value - low_value)).astype(np.float32)
#         image_array = image_array * 2 - 1
#         image = np.expand_dims(image_array, axis=2)
#         return image, min_value, max_value
#
#
#     elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # 如果输入的是三通道数据
#         # 定义无数据值集合（包含-32767、-32768和极小数）
#         nodata_values = {-32767, -32768, -3.4028234663852886e+38}
#         nodata_list = list(nodata_values)
#
#         min_values = np.min(image_array, axis=(0, 1))  # 初始化为全量数据统计（后续会被覆盖）
#         max_values = np.max(image_array, axis=(0, 1))
#         image_size, _, num_channels = image_array.shape
#
#         # 重新计算各通道有效数据的统计量
#         for i in range(num_channels):
#             channel = image_array[:, :, i].astype(np.float32)
#             valid_mask = np.ones_like(channel, dtype=bool)
#             for nodata in nodata_list:
#                 valid_mask &= (channel != nodata)
#             valid_data = channel[valid_mask]
#             if valid_data.size > 0:
#                 min_values[i] = np.nanmin(valid_data)
#                 max_values[i] = np.nanmax(valid_data)
#
#         ranges = max_values - min_values
#         low_values = min_values - ranges * 0.1
#         high_values = max_values + ranges * 0.1
#
#         normalized = np.zeros_like(image_array, dtype=np.float32)
#         for i in range(num_channels):
#             channel = image_array[:, :, i].astype(np.float32)
#             valid_mask = np.ones_like(channel, dtype=bool)
#             for nodata in nodata_list:
#                 valid_mask &= (channel != nodata)
#             scale = high_values[i] - low_values[i]
#             scale = scale if scale != 0 else 1.0
#             normalized[:, :, i] = ((channel - low_values[i]) / scale) * 2 - 1
#             normalized[~valid_mask, i] = -2.0
#
#         return normalized, min_values, max_values
#
#     # elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # 如果输入的是三通道数据
#     #
#     #     min_values = np.min(image_array, axis=(0, 1))  # 计算每个通道的最小值
#     #     max_values = np.max(image_array, axis=(0, 1))  # 计算每个通道的最大值
#     #     ranges = max_values - min_values
#     #     low_values = min_values - ranges * 0.1
#     #     high_values = max_values + ranges * 0.1
#     #
#     #     image_size, _, num_channels = image_array.shape
#     #
#     #     for i in range(num_channels):
#     #         image_array[:, :, i] = (
#     #                 (image_array[:, :, i] - low_values[i]) / (high_values[i] - low_values[i])).astype(
#     #             np.float32)
#     #         image_array[:, :, i] = image_array[:, :, i] * 2 - 1
#     #     image_array[np.isnan(image_array)] = 0
#     #     return image_array, min_values, max_values
#     else:
#         print("the shape of image is error!")


# 文件: image_datasets.py

def dem_zip(image_array):  # 将dem像素值压缩
    # 将哨兵2中的nan值替换成0

    if len(image_array.shape) == 2:  # 如果输入的是单通道数据

        # 无数据值不参与归一化
        valid_mask = image_array != -32767
        valid_data = image_array[valid_mask]

        # --- START OF MODIFICATION 1 ---
        # 增加对valid_data为空的检查
        if valid_data.size == 0:
            # 如果所有像素都是nodata，直接返回一个全为-1的数组（或你认为合适的默认值）
            image_array = np.full_like(image_array, -1.0, dtype=np.float32)
            image = np.expand_dims(image_array, axis=2)
            return image, -1, -1  # 返回默认的min/max

        min_value = np.nanmin(valid_data)
        max_value = np.nanmax(valid_data)
        # --- END OF MODIFICATION 1 ---

        data_range = max_value - min_value
        low_value = min_value - data_range * 0.1
        high_value = max_value + data_range * 0.1

        # --- START OF MODIFICATION 2 ---
        # 增加对分母为零的检查
        scale = high_value - low_value
        if scale == 0:
            image_array = np.zeros_like(image_array, dtype=np.float32)  # 所有值相同，归一化后为常数，这里设为0
        else:
            image_array = ((image_array - low_value) / scale).astype(np.float32)
            image_array = image_array * 2 - 1
        # --- END OF MODIFICATION 2 ---

        image = np.expand_dims(image_array, axis=2)
        return image, min_value, max_value


    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # 如果输入的是三通道数据
        # 定义无数据值集合（包含-32767、-32768和极小数）
        nodata_values = {-32767, -32768, -3.4028234663852886e+38}
        nodata_list = list(nodata_values)

        min_values = np.zeros(3, dtype=np.float32)  # 初始化为0
        max_values = np.zeros(3, dtype=np.float32)  # 初始化为0
        image_size, _, num_channels = image_array.shape

        # 重新计算各通道有效数据的统计量
        for i in range(num_channels):
            channel = image_array[:, :, i].astype(np.float32)
            valid_mask = np.ones_like(channel, dtype=bool)
            for nodata in nodata_list:
                valid_mask &= (channel != nodata)
            valid_data = channel[valid_mask]
            if valid_data.size > 0:
                min_values[i] = np.nanmin(valid_data)
                max_values[i] = np.nanmax(valid_data)
            else:
                # 如果整个通道都是nodata, 给定一个默认值
                min_values[i] = 0
                max_values[i] = 0

        ranges = max_values - min_values
        low_values = min_values - ranges * 0.1
        high_values = max_values + ranges * 0.1

        normalized = np.zeros_like(image_array, dtype=np.float32)
        for i in range(num_channels):
            channel = image_array[:, :, i].astype(np.float32)
            valid_mask = np.ones_like(channel, dtype=bool)
            for nodata in nodata_list:
                valid_mask &= (channel != nodata)

            # --- START OF MODIFICATION 3 (The Core Fix) ---
            scale = high_values[i] - low_values[i]
            if scale == 0:
                # 如果scale为0，说明这个通道的所有有效值都相同。
                # 归一化后应该是一个常数，这里我们设为0 (对应[-1, 1]的中心)。
                normalized[:, :, i] = 0.0
            else:
                # 只有当scale不为0时，才进行除法运算
                normalized[:, :, i] = ((channel - low_values[i]) / scale) * 2 - 1
            # --- END OF MODIFICATION 3 ---

            normalized[~valid_mask, i] = -2.0  # -2.0 是一个可以考虑的值，或者用-1.0

        return normalized, min_values, max_values

    else:
        print("the shape of image is error!")
        # 最好在这里抛出异常而不是只打印
        raise ValueError(f"Unsupported image shape: {image_array.shape}")