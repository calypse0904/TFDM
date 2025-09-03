"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
# 导入所需的库
import argparse  # 用于解析命令行参数
import os  # 用于处理操作系统功能，如文件路径
import torch as th  # 导入 PyTorch 库，并重命名为 th
import yaml  # 用于处理 YAML 配置文件
from guided_diffusion import dist_util  # 从 guided_diffusion 包中导入 dist_util，用于分布式计算相关的功能
from guided_diffusion.image_datasets import load_data_inpa, eval_imswrite  # 导入图像数据集相关功能
from guided_diffusion.recordImages import writeRecords  # 导入记录图像的功能
import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')

# Workaround
try:
    import ctypes  # 尝试导入 ctypes 库，该库用于与 C 语言代码进行交互。

    libgcc_s = ctypes.CDLL('libgcc_s.so.1')  # 使用 ctypes 加载名为 'libgcc_s.so.1' 的共享库。该库通常是 GCC（GNU 编译器）的一部分，可能用于处理异常或其他低级任务。
except:
    pass  # 如果导入 ctypes 或加载库时发生异常，捕获异常并继续执行，不做处理。

# 导入 guided_diffusion 中的功能函数
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,  # 导入默认的模型和扩散过程配置（可能用于配置模型的参数或训练设置）。
    create_model_and_diffusion,   # 导入创建模型和扩散过程的函数，可能用于初始化模型和相关的扩散过程。
    sr_model_and_diffusion_defaults,  # 导入超分辨率模型和扩散过程的默认配置。
    sr_create_model_and_diffusion,    # 导入超分辨率模型和扩散过程的创建函数。
    select_args,   # 导入选择参数的函数，可能用于解析命令行参数或配置文件中的设置。
)  # noqa: E402  # noqa: E402 表示跳过 PEP8 中关于模块导入顺序的警告。


def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)  # 将sample从[-1,1]转换到[0,255]，并保证数据是uint8类型
    sample = sample.permute(0, 2, 3,
                            1)  # 将tensor数据的维度从[batch_size, num_channels, height, width]变为[batch_size, height, width, num_channels]
    sample = sample.contiguous()  # 返回一个内存连续的tensor，提高计算效率
    sample = sample.detach().cpu().numpy()  # 将结果转换成numpy，并从GPU转移到CPU上
    return sample[:, :, :, 0]


def to255(sample):
    if sample is None:
        return sample
    sample = ((sample + 1) / 2 * 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample[0, :, :, 0]


def toF32(sample, min_e, max_e):
    if sample is None:
        return sample
    sample = (sample + 1) / 2
    range = max_e - min_e
    low_value = min_e - range * 0.1
    high_value = max_e + range * 0.1
    sample = (high_value - low_value) * sample + low_value

    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    img = sample[:, :, :, 0]
    return img


def writeRecordX0(records, times, min_e, max_e, imageNames, path):
    index = 0;
    imgs = []
    pre = times[0]
    for record, time in zip(records, times):
        if index % 200 == 0 or index < 10 or time == 0:
            if time < pre:
                img32 = toF32(record, min_e, max_e)
                imgName = r'time' + str(index) + "_" + str(time) + "_" + imageNames[0]
                imgName_list = []
                imgName_list.append(imgName)
                # write_images(img32,imgName_list,path)

                img8 = to255(record)
                imgs.append(img8)

                pre = time
        index += 1
    writeRecords(imgs, path, imageNames[0])


def main(conf):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 设置CUDA设备为0，确保程序只使用设备0进行计算
    device = dist_util.dev()  # 获取设备  # 获取可用的计算设备（GPU或CPU），调用dist_util.dev()来确定是使用GPU还是CPU
    conf.device = device
    dict_model_and_diffusion = {**vars(conf.model), **(vars(conf.diffusion))}   # 合并模型和扩散模型的配置，生成一个字典
    if conf.condition.condition_flag == "uncondition":
        model, diffusion = create_model_and_diffusion(
            **select_args(dict_model_and_diffusion, model_and_diffusion_defaults().keys()), conf=conf
            # 使用conf中的参数值，给model_and_diffusion_defaults中的参数赋值
        )
    else:
        dict_model_and_diffusion.update(**vars(conf.condition))
        model, diffusion = sr_create_model_and_diffusion(
            **select_args(dict_model_and_diffusion, sr_model_and_diffusion_defaults()), conf=conf
        )
        # summary(model.cuda(device=device),[(1,256,256),(1,),(1,256,256)],1)
    model.load_state_dict(  # 从conf.model_path指定的checkpoint加载模型
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.model.use_fp16:  # 判断是否将一部分数据转换成半浮点数，节省空间占用，提高模型效率
        model.convert_to_fp16()
    model.eval()  # 设置模型为评估模式

    show_progress = conf.show_progress  # 这个参数是用来显示进度条

    cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        return model(x, t, y, gt=gt)

    print("sampling...")
    all_images = []

    dl = load_data_inpa(conf, model_flag='eval')  # 获取数据加载器

    for batch in iter(dl):  # 读取每个batch的数据,dem的min_e,max_e都保存在了batch里

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']  # 将GT对应的tensor保存到model_kwargs，对应键值gt
        if conf.condition.condition_flag != "uncondition":
            model_kwargs["y"] = batch['refer']
        gt_keep_mask = batch.get('gt_keep_mask')  # 获取掩膜
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        sample_fn = (  # 使用扩散模型的递进采样
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        result, ref, records, times = sample_fn(
            model_fn,
            (batch_size, conf.model.in_channels, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        """
        srs = toU8(result['sample'])#从[-1,1]解压到[0,255]
        gts = toU8(result['gt'])
        #th.ones_like(result.get('gt'))生成一个和gt形状一样的全1张量
        #+后半段，将空洞内部的像素值设置为-1
        #最终的lrs，保留空洞外部的图像，空洞内部的结果为-1
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))#背景为1，空洞为-1
        """

        min_e = batch['gt_min'].item()
        max_e = batch['gt_max'].item()
        writeRecordX0(records, times, min_e, max_e, batch['GT_name'], path=conf.data.eval.paths.records)

        srs = toF32(result['sample'], min_e=min_e, max_e=max_e)
        gts = toF32(result['gt'], min_e=min_e, max_e=max_e)
        tmp = result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) * \
              th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask'))
        lrs = toF32(tmp, min_e=min_e, max_e=max_e)
        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))  # 背景为1，空洞为-1

        eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], conf=conf, verify_same=False)

    print("sampling complete")


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = parser.parse_args()  # 将命令行参数转换成字典，key对应value
    with open(args.conf_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    main(new_config)
