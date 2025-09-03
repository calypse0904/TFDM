"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum

import numpy as np
import torch as th
import math

from collections import defaultdict

from guided_diffusion.scheduler import get_schedule_jump
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, use_scale):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.

        if use_scale:
            scale = 1000 / num_diffusion_timesteps
        else:
            scale = 1

        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        conf=None
    ):
        self.model_mean_type = model_mean_type#均值类型，三种情况：直接预测均值；预测x_0；预测噪声
        self.model_var_type = model_var_type#方差类型：固定方差，预测方差（直接预测，预测范围）
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.conf = conf

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)#\bar{alpha}
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])#在\bar{alpha}的前面加上1.0，挤掉最后一个元素
        self.alphas_cumprod_prev_prev = np.append(#再用一个1.0挤掉最后一个元素
            1.0, self.alphas_cumprod_prev[:-1])

        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)#在\bar{alpha}最后补上0.0，舍弃第一个元素

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)#对\bar{alpha}求根号
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)#根号（1-\bar{alpha}
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)#log(1-\bar{alpha})
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)#根号\bar{alpha}的倒数
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (#公式11中的第一个系数
            betas * np.sqrt(self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (#公式11中的第二个系数
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = _extract_into_tensor(self.betas, t, img_out.shape)

        img_in_est = th.sqrt(1 - beta) * img_out + \
            th.sqrt(beta) * th.randn_like(img_out)

        return img_in_est
    def get_ref_noised(self,ref_gt,conf,t):#定义低分辨率参考图像的加噪结果
        alpha_cumprod = _extract_into_tensor(
            self.alphas_cumprod, t, ref_gt.shape)
        ref_weight = th.sqrt(alpha_cumprod)  # 计算张量的平方根
        ref_part = ref_weight * ref_gt

        noise_weight = th.sqrt((1 - alpha_cumprod))
        noise_part = noise_weight * th.randn_like(ref_gt)

        weighed_ref = ref_part + noise_part
        return weighed_ref
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1,
                                 t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2,
                                   t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        model就是UNet，x是x_t，得到t-1时刻的mean和variance，
        得到扩散过程的均值和方差，也包括x_start的预测
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        tmp={**model_kwargs}
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)#UNet接受x_t和时刻t作为输入，model_output取决于模型的训练方式

        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)#分割model的output

        #得到方差和对数方差，这里是repaint进行的修改，在guided diffusion中，保留了固定方差的实现，这里只使用可学习方差
        if self.model_var_type == ModelVarType.LEARNED:#直接预测方差
            model_log_variance = model_var_values#模型输出的是对数方差
            model_variance = th.exp(model_log_variance)#对数方差取指数得到模型预测的方差
        else:#预测方差的范围，IDDPM的改进
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )#\bar{β_t}的对数
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)#β_t的对数
            frac = (model_var_values + 1) / 2#模型预测的范围[-1,1]，转换成[0,1]
            model_log_variance = frac * max_log + (1 - frac) * min_log#frac就是公式15中的v
            model_variance = th.exp(model_log_variance)#取指数

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)#如果denoised_fn不是空的，那么就对x做去噪操作
            if clip_denoised:
                return x.clamp(-1, 1)#将张量限制在[-1,1]的范围
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:#case1:预测x_{t-1}的期望值，也就是直接预测均值\mu_t
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )#在训练中用不到，但是在evaluation要用到，就是用x_{t-1},x_t,mean，对应公式11，12算出x_0
            model_mean = model_output#模型的output就是均值
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:#case2:如果模型是预测x_0或噪声epsilon
            if self.model_mean_type == ModelMeanType.START_X:#预测x_0
                pred_xstart = process_xstart(model_output)#后处理函数，获得x_0
            else:#case3:预测噪声epsilon
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)#从x_t,t,eps推断出x_0，对应公式9
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )#根据公式11计算均值
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """

        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)


        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] *
            gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(#基于x_t采样出x_[t-1}
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        conf=None,
        meas_fn=None,
        pred_xstart=None,
        idx_wall=-1
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        noise = th.randn_like(x)#从标准正态分布中采样，得到一个和x形状一样的噪声张量

        if conf.inpa_inj_sched_prev:#这里也是和guided diffusion不同的地方，额外加的部分

            if pred_xstart is not None:#pre_xstart是上一步去噪过程预测的x_0
                gt_keep_mask = model_kwargs.get('gt_keep_mask')
                if gt_keep_mask is None:
                    gt_keep_mask = conf.get_inpa_mask(x)

                gt = model_kwargs['gt']

                alpha_cumprod = _extract_into_tensor(
                    self.alphas_cumprod, t, x.shape)

                if conf.inpa_inj_sched_prev_cumnoise:
                    weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
                else:#其实就是原图部分加噪，利用x_0得到x_t。
                    gt_weight = th.sqrt(alpha_cumprod)#计算张量的平方根
                    gt_part = gt_weight * gt

                    noise_weight = th.sqrt((1 - alpha_cumprod))
                    noise_part = noise_weight * th.randn_like(x)

                    weighed_gt = gt_part + noise_part

                x = (#拼接原图和x_t
                    gt_keep_mask * (
                        weighed_gt
                    )
                    +
                    (1 - gt_keep_mask) * (
                        x
                    )
                )

        # 得到x_{t-1}的均值、方差、对数方差，x_0的预测值
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        #非0时刻的掩码矩阵
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        #和类引导有关的
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        #重参数化，采样出x_{t-1}
        sample = out["mean"] + nonzero_mask * \
            th.exp(0.5 * out["log_variance"]) * noise#模型预测的均值+方差*噪音

        result = {"sample": sample,
                  "pred_xstart": out["pred_xstart"], 'gt': model_kwargs.get('gt')}

        return result

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        records=[]
        times=[]
        for sample,time in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            conf=conf
        ):
            final = sample
            records.append(sample["pred_xstart"])
            times.append(time)
        gt_keep_mask = model_kwargs.get('gt_keep_mask')
        gt=model_kwargs.get('gt')
        ref=model_kwargs.get('y')
        final['sample']=(gt_keep_mask*gt)+(1-gt_keep_mask)*final.get('sample')
        if return_all:
            return final,ref,records,times
        else:
            return final["sample"],records,times

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conf=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:#如果noise，也就是T时刻的噪音为空，则随机采样一个噪音image_after_step
            image_after_step = th.randn(*shape, device=device)#从标准正态分布随机采样出一个NCHW的张量

        self.gt_noises = None  # reset for next image


        pred_xstart = None

        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)#创建一个默认值为0的字典

        if conf.schedule_jump_params:
            schedule_params=vars(conf.schedule_jump_params)
            times = get_schedule_jump(**schedule_params)#就是以10次连续加噪，10次连续去噪的操作为一个单元，重复10个单元得到下一个阶段，对应论文中的反复迭代使得语义一致

            time_pairs = list(zip(times[:-1], times[1:]))#将相邻时间步组成元组
            if progress:
                from tqdm.auto import tqdm
                time_pairs = tqdm(time_pairs)

            for t_last, t_cur in time_pairs:
                idx_wall += 1
                t_last_t = th.tensor([t_last] * shape[0],  # pylint: disable=not-callable
                                     device=device)#t_last_t是一个形状为batch_size的张量，每个元素都是t_last的值

                if t_cur < t_last:  # reverse，去噪过程
                    with th.no_grad():#禁止梯度计算，提高计算效率
                        image_before_step = image_after_step.clone()
                        out = self.p_sample(#采样出x_{t-1}
                            model,
                            image_after_step,
                            t_last_t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            cond_fn=cond_fn,
                            model_kwargs=model_kwargs,
                            conf=conf,
                            pred_xstart=pred_xstart
                        )
                        image_after_step = out["sample"]
                        pred_xstart = out["pred_xstart"]

                        sample_idxs[t_cur] += 1

                        yield out,t_last#返回out的值，但不终止函数的执行

                else:#加噪过程
                    t_shift = 1

                    image_before_step = image_after_step.clone()
                    image_after_step = self.undo(#一次加噪
                        image_before_step, image_after_step,
                        est_x_0=out['pred_xstart'], t=t_last_t+t_shift, debug=False)
                    pred_xstart = out["pred_xstart"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(#真实的均值和方差，都是x_{t-1}的均值和方差
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(#模型预测的均值和方差，同样是x_{t-1}
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        # --- START OF MODIFICATION ---
        # 在使用模型预测的对数方差之前，对其进行范围限制，以增强数值稳定性。
        # 这可以防止log(variance)趋近于-inf，从而避免在后续计算中产生nan。
        out["log_variance"] = out["log_variance"].clamp(min=-30, max=20)
        # --- END OF MODIFICATION ---


        kl = normal_kl(#两个高斯分布的方差
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)#前向过程采样出x_t

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            #如果模型采用MSE loss，需要根据模型预测类型进行判断
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:#case1:预测方差
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)# frozen_out 是让方差的学习不影响均值的优化
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)#torch.mean
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
