
import os
import torch as th
import copy
import functools
import blobfile as bf
from torch.optim import AdamW
from guided_diffusion import logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema

from guided_diffusion.resample import UniformSampler, LossAwareSampler
def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None
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
def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv(key, values.mean().item())
        # logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv(f"{key}_q{quartile}", sub_loss)
            # logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

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
        train_loss_dict,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        model_name,
        save_dir,
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
        self.train_loss_dict = train_loss_dict
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        self.timestamp = model_name
        self.save_dir = save_dir

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(th.load(resume_checkpoint, map_location=self.device))

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(ema_checkpoint, map_location=self.device)
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = os.path.join(os.path.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt")
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=self.device)
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch,label= next(self.data)
            train_loss =self.run_step(batch,label)
            if self.step % self.log_interval == 0:
                self.log_step()
                self.train_loss_dict[self.step] = train_loss
                logger.dumpkvs()

            if (
                self.step
                and self.step % self.save_interval == 0
            ):
                self.save()
                # Run for a finite amount of time in integration tests.
                # if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                #     return
            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        return self.train_loss_dict

    def run_step(self, batch,label):
        batch_losses =self.forward_backward(batch,label)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        return batch_losses


    def forward_backward(self, batch,cond_label):
        self.mp_trainer.zero_grad()
        batch = batch.to(self.device)
        cond_label = {k: v.to(self.device) for k, v in cond_label.items()}
        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)


        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            batch,
            t,
            model_kwargs=cond_label,
        )

        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()

        log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

        self.mp_trainer.backward(loss)

        return loss.item()


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
        logger.logkv("Att:train_step", self.step + self.resume_step)
        #logger.logkv("samples", (self.step + self.resume_step + 1) * self.batch_size)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(self.save_dir, self.timestamp, filename), "wb") as f:
                th.save(state_dict, f)

        if not os.path.exists(os.path.join(self.save_dir, self.timestamp)):
            os.mkdir(os.path.join(self.save_dir, self.timestamp))
        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        with bf.BlobFile(bf.join(self.save_dir, self.timestamp, f"opt{(self.step+self.resume_step):06d}.pt"), "wb") as f:
            th.save(self.opt.state_dict(), f)
