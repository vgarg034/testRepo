import torch
from tqdm import tqdm
from ..utils import AverageMeter

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    _xla_available = True
except ImportError:
    _xla_available = False
try:
    from apex import amp

    _apex_available = True
except ImportError:
    _apex_available = False


def reduce_fn(vals):
    return sum(vals) / len(vals)


class Engine:
    @staticmethod
    def train(
        data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1,
        use_tpu=False,
        fp16=False,
    ):
        if use_tpu and not _xla_available:
            raise Exception(
                "You want to use TPUs but you dont have pytorch_xla installed"
            )
        if fp16 and not _apex_available:
            raise Exception("You want to use fp16 but you dont have apex installed")
        if fp16 and use_tpu:
            raise Exception("Apex fp16 is not available when using TPUs")
        if fp16:
            accumulation_steps = 1
        losses = AverageMeter()
        predictions = []
        model.train()
        if accumulation_steps > 1:
            optimizer.zero_grad()
        if use_tpu:
            para_loader = pl.ParallelLoader(data_loader, [device])
            tk0 = tqdm(
                para_loader.per_device_loader(device),
                total=len(data_loader),
                desc=f"[xla:{xm.get_ordinal()}]",
            )
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))

        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            if accumulation_steps == 1 and b_idx == 0:
                optimizer.zero_grad()
            _, loss = model(**data)

            if not use_tpu:
                with torch.set_grad_enabled(True):
                    if fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    if (b_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        if b_idx > 0:
                            optimizer.zero_grad()
            else:
                loss.backward()
                xm.optimizer_step(optimizer)
                if scheduler is not None:
                    scheduler.step()
                if b_idx > 0:
                    optimizer.zero_grad()
            if use_tpu:
                reduced_loss = xm.mesh_reduce("loss_reduce", loss, reduce_fn)
                losses.update(reduced_loss.item(), data_loader.batch_size)
            else:
                losses.update(loss.item(), data_loader.batch_size)

            tk0.set_postfix(loss=losses.avg)
        return losses.avg

    @staticmethod
    def evaluate(data_loader, model, device, use_tpu=False):
        losses = AverageMeter()
        model.eval()
        with torch.no_grad():
            if use_tpu:
                para_loader = pl.ParallelLoader(data_loader, [device])
                tk0 = tqdm(
                    para_loader.per_device_loader(device),
                    total=len(data_loader),
                    desc=f"[xla:{xm.get_ordinal()}]",
                )
            else:
                tk0 = tqdm(data_loader, total=len(data_loader))
            for data in tk0:
                for key, value in data.items():
                    data[key] = value.to(device)
                _, loss = model(**data)
                if use_tpu:
                    reduced_loss = xm.mesh_reduce("loss_reduce", loss, reduce_fn)
                    losses.update(reduced_loss.item(), data_loader.batch_size)
                else:
                    losses.update(loss.item(), data_loader.batch_size)
                tk0.set_postfix(loss=losses.avg)
        return losses.avg
