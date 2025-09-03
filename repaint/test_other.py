"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from utils import imwrite
from guided_diffusion import dist_util
from guided_diffusion.image_datasets import load_data_inpa, eval_imswrite
import yaml
# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    select_args,
)  # noqa: E402

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)#将sample从[-1,1]转换到[0,255]，并保证数据是uint8类型
    sample = sample.permute(0, 2, 3, 1)#将tensor数据的维度从[batch_size, num_channels, height, width]变为[batch_size, height, width, num_channels]
    sample = sample.contiguous()#返回一个内存连续的tensor，提高计算效率
    sample = sample.detach().cpu().numpy()#将结果转换成numpy，并从GPU转移到CPU上
    return sample

def toF32(sample,min_e,max_e):
    if sample is None:
        return sample
    sample=(sample+1)/2
    range=max_e-min_e
    low_value=min_e-range*0.1
    high_value=max_e+range*0.1
    sample=(high_value-low_value)*sample+low_value

    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    img=sample[:,:,:,0]
    return img

def main(conf):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = dist_util.dev()#获取设备
    conf.device=device
    dict_model_and_diffusion={**vars(conf.model),**(vars(conf.diffusion))}
    if conf.condition.condition_flag=="uncondition":
        model, diffusion = create_model_and_diffusion(
            **select_args(dict_model_and_diffusion, model_and_diffusion_defaults().keys()), conf=conf#使用conf中的参数值，给model_and_diffusion_defaults中的参数赋值
        )
    else:
        dict_model_and_diffusion.update(**vars(conf.condition))
        model,diffusion=sr_create_model_and_diffusion(
            **select_args(dict_model_and_diffusion,sr_model_and_diffusion_defaults()),conf=conf
        )
    model.load_state_dict(#从conf.model_path指定的checkpoint加载模型
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.model.use_fp16:#判断是否将一部分数据转换成半浮点数，节省空间占用，提高模型效率
        model.convert_to_fp16()
    model.eval()#设置模型为评估模式

    show_progress = conf.show_progress#这个参数是用来显示进度条

    cond_fn = None
    def model_fn(x, t, y=None, gt=None, **kwargs):
        return model(x, t, y, gt=gt)

    print("sampling...")
    all_images = []

    dl = load_data_inpa(conf,model_flag='eval')#获取数据加载器

    for batch in iter(dl):#读取每个batch的数据,dem的min_e,max_e都保存在了batch里

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']#将GT对应的tensor保存到model_kwargs，对应键值gt
        if conf.condition.condition_flag!="uncondition":
            model_kwargs["y"]=batch['refer']
        gt_keep_mask = batch.get('gt_keep_mask')#获取掩膜
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        sample_fn = (#使用扩散模型的递进采样
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )


        result,ref,records,times = sample_fn(
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
        min_e=batch['gt_min'].item()
        max_e=batch['gt_max'].item()
        srs=toF32(result['sample'],min_e=min_e,max_e=max_e)
        gts=toF32(result['gt'],min_e=min_e,max_e=max_e)
        tmp=result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *\
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask'))
        lrs = toF32(tmp,min_e=min_e,max_e=max_e)
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
    args = parser.parse_args()#将命令行参数转换成字典，key对应value
    with open(args.conf_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    main(new_config)
