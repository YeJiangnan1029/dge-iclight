import sys
import warnings
from bisect import bisect_right

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import threestudio

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips

from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import torch.nn.functional as F

def compute_metrics(img1, img2, lpips_model=None):
    
    """
    输入:
        img1, img2: Tensor，形状为 (3, H, W)，值域为 [0, 1]，图像范围为 RGB
        lpips_model: 已加载的 LPIPS 模型（可复用）
    返回:
        dict: {psnr, ssim, lpips}
    """
    assert img1.shape == img2.shape, "Input images must have the same shape"
    if img1.ndim == 3 and img1.shape[2] == 3:
        img1 = img1.permute(2,0,1)
        img2 = img2.permute(2,0,1)
    assert img1.ndim == 3 and img1.shape[0] == 3, "Images must be in (3, H, W) format"

    # --- PSNR & SSIM (convert to numpy, range [0, 1]) ---
    img1_np = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.detach().cpu().numpy().transpose(1, 2, 0)

    psnr = compare_psnr(img1_np, img2_np, data_range=1.0)
    ssim = compare_ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)

    # --- LPIPS (input shape: NCHW, range: [-1, 1]) ---
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net='alex').eval().to(img1.device)

    img1_lpips = img1[None] * 2 - 1  # (1, 3, H, W)
    img2_lpips = img2[None] * 2 - 1

    lpips_val = lpips_model(img1_lpips, img2_lpips).item()

    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_val}

# 加载 CLIP 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def clip_preprocess_tensor(tensor: torch.Tensor):
    # tensor shape: (1, 3, H, W)
    if tensor.shape[1] != 3:
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[-1] != 224 or tensor.shape[-2] != 224:
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
    # Normalize
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std

def compute_clip_metrics(img_before: torch.Tensor, img_after: torch.Tensor, text: str, neutral_text="a photo"):
    """
    img_before, img_after: Tensor of shape (1, 3, H, W), range [0,1]
    """
    img_before = clip_preprocess_tensor(img_before)
    img_after = clip_preprocess_tensor(img_after)

    # Get CLIP features
    image_emb_before = clip_model.get_image_features(img_before)
    image_emb_after = clip_model.get_image_features(img_after)

    # Text encoding
    text_inputs = clip_processor(text=[text, neutral_text], return_tensors="pt", padding=True).to(device)
    text_embs = clip_model.get_text_features(**text_inputs)
    text_emb, neutral_emb = text_embs[0], text_embs[1]

    # Normalize
    image_emb_before = F.normalize(image_emb_before, dim=-1)
    image_emb_after = F.normalize(image_emb_after, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    neutral_emb = F.normalize(neutral_emb, dim=-1)

    # CLIP similarity
    clip_sim = torch.cosine_similarity(image_emb_after, text_emb, dim=-1).item()

    # Directional similarity
    image_dir = image_emb_after - image_emb_before
    text_dir = text_emb - neutral_emb
    dir_sim = F.cosine_similarity(image_dir, text_dir, dim=-1).item()

    return clip_sim, dir_sim

def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError


def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model):
    if hasattr(config, "params"):
        params = [
            {"params": get_parameters(model, name), "name": name, **args}
            for name, args in config.params.items()
        ]
        threestudio.debug(f"Specify optimizer params: {config.params}")
    else:
        params = model.parameters()
    if config.name in ["FusedAdam"]:
        import apex

        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    elif config.name in ["Adan"]:
        from threestudio.systems import optimizers

        optim = getattr(optimizers, config.name)(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler_to_instance(config, optimizer):
    if config.name == "ChainedScheduler":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.ChainedScheduler(schedulers)
    elif config.name == "Sequential":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=config.milestones
        )
    else:
        scheduler = getattr(lr_scheduler, config.name)(optimizer, **config.args)
    return scheduler


def parse_scheduler(config, optimizer):
    interval = config.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        scheduler = {
            "scheduler": lr_scheduler.SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        scheduler = {
            "scheduler": lr_scheduler.ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(config.name)(optimizer, **config.args),
            "interval": interval,
        }
    return scheduler
