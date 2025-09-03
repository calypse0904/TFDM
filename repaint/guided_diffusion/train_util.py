import copy
import functools
import os

import blobfile as bf
import torch
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(#创建混合精度训练器
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(#创建优化器
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():#并行训练
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch,cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    # def forward_backward(self, batch, cond):
    #     self.mp_trainer.zero_grad()#梯度清零
    #     for i in range(0, batch.shape[0], self.microbatch):#将一个batch划分成多个小的batch，在每个小batch上进行前向传播和反向传播，最后将多个小batch的梯度进行累加，更新模型参数
    #         micro = batch[i : i + self.microbatch].to(dist_util.dev())
    #         micro_cond = {
    #             k: v[i : i + self.microbatch].to(dist_util.dev())
    #             for k, v in cond.items()
    #         }
    #
    #         last_batch = (i + self.microbatch) >= batch.shape[0]
    #         t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())#重要性采样，返回的是选中时间步下标t,权重weights
    #
    #         compute_losses = functools.partial(#生成一个新的函数compute_losses，便于多次调用
    #             self.diffusion.training_losses,
    #             self.ddp_model,
    #             micro,
    #             t,
    #             model_kwargs=micro_cond,
    #         )#这些定义的参数都是固定参数
    #
    #         if last_batch or not self.use_ddp:
    #             losses = compute_losses()
    #         else:
    #             with self.ddp_model.no_sync():
    #                 losses = compute_losses()
    #
    #
    #         if isinstance(self.schedule_sampler, LossAwareSampler):
    #             self.schedule_sampler.update_with_local_losses(
    #                 t, losses["loss"].detach()
    #             )
    #
    #         loss = (losses["loss"] * weights).mean()
    #         log_loss_dict(
    #             self.diffusion, t, {k: v * weights for k, v in losses.items()}
    #         )
    #         self.mp_trainer.backward(loss)#其实就是调用Loss的backward

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()  # 梯度清零
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            # --- 调试点 1: 检查最原始的输入数据 ---
            if torch.isnan(micro).any() or torch.isinf(micro).any():
                print(f"!!!!!! [调试点1] 崩溃！在 step {self.step}，输入模型的 'micro' (主数据) 中发现 NaN/Inf。")
                print(f"!!!!!! 这意味着数据加载或归一化步骤出了问题。")
                raise ValueError("输入数据 'micro' 损坏")

            if "low_res" in micro_cond and (
                    torch.isnan(micro_cond["low_res"]).any() or torch.isinf(micro_cond["low_res"]).any()):
                print(f"!!!!!! [调试点1] 崩溃！在 step {self.step}，输入模型的 'micro_cond' (条件数据) 中发现 NaN/Inf。")
                raise ValueError("输入数据 'micro_cond' 损坏")

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            # --- 调试点 2: 检查损失函数计算的直接输出 ---
            for loss_name, loss_value in losses.items():
                if torch.isnan(loss_value).any() or torch.isinf(loss_value).any():
                    print(f"!!!!!! [调试点2] 崩溃！在 step {self.step}，函数 training_losses 返回的 '{loss_name}' 包含 NaN/Inf。")
                    print(f"!!!!!! 这意味着问题出在 guided_diffusion/gaussian_diffusion.py 的 losses 计算中。")
                    # 可以在这里保存导致问题的输入，以便离线分析
                    # torch.save(micro, f'bad_micro_step_{self.step}.pt')
                    # torch.save(micro_cond, f'bad_cond_step_{self.step}.pt')
                    # torch.save(t, f'bad_t_step_{self.step}.pt')
                    raise ValueError(f"损失函数计算失败，'{loss_name}' 损坏")

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            # --- 调试点 3: 检查最终加权后的总loss ---
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"!!!!!! [调试点3] 崩溃！在 step {self.step}，最终加权后的总 'loss' 变成了 NaN/Inf。")
                print(f"!!!!!! 这可能是由于 'weights' 包含了 NaN/Inf，或者 losses['loss'] 已经是 NaN/Inf。")
                raise ValueError("最终 'loss' 损坏")

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):#zip是匹配时间步ts和值values两个张量对应位置的元素
            quartile = int(4 * sub_t / diffusion.num_timesteps)#其实就是重要性采样的步数放大到[0,4]，最后打印的vb_q1,q2之类的1，2都是quartile
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
