import argparse
import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')
import conf_mgt
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_inpa
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    select_args
)
from guided_diffusion.train_util import TrainLoop
import yaml
import torch

def main(conf):

    logger.configure(dir=conf.train.save_path)
    dist_util.setup_dist()
    logger.log("creating model and diffusion...")
    device = dist_util.dev()  # 获取设备
    conf.device = device
    dict_model_and_diffusion = {**vars(conf.model), **(vars(conf.diffusion))}
    if conf.condition.condition_flag=="uncondition":
        model, diffusion = create_model_and_diffusion(
            **select_args(dict_model_and_diffusion, model_and_diffusion_defaults().keys()), conf=conf#使用conf中的参数值，给model_and_diffusion_defaults中的参数赋值
        )
        data=load_uncondition_data(conf)
    else:
        dict_model_and_diffusion.update(**vars(conf.condition))
        model,diffusion=sr_create_model_and_diffusion(
            **select_args(dict_model_and_diffusion,sr_model_and_diffusion_defaults()),conf=conf
        )
        data = load_superres_data(conf)
    model.to(conf.device)
    schedule_sampler = create_named_schedule_sampler(conf.train.schedule_sampler, diffusion)

    logger.log("creating data loader...")


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=conf.train.batch_size,
        microbatch=conf.train.microbatch,
        lr=conf.train.lr,
        ema_rate=conf.train.ema_rate,
        log_interval=conf.train.log_interval,
        save_interval=conf.train.save_interval,
        resume_checkpoint=conf.train.resume_checkpoint,
        use_fp16=conf.model.use_fp16,
        fp16_scale_growth=conf.train.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=conf.train.weight_decay,
        lr_anneal_steps=conf.train.lr_anneal_steps,
    ).run_loop()

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def load_superres_data(conf):
    data = load_data_inpa(conf=conf,model_flag='train')
    for large_batch in data:
        model_kwargs={}
        model_kwargs["low_res"] =large_batch['refer']
        yield large_batch['GT'], model_kwargs

def load_uncondition_data(conf):
    data=load_data_inpa(conf=conf,model_flag='train')
    for large_batch in data:
        model_kwargs={}
        yield large_batch['GT'],model_kwargs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = parser.parse_args()  # 将命令行参数转换成字典，key对应value
    with open(args.conf_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    main(new_config)
