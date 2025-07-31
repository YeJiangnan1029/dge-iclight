from dataclasses import dataclass
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
import torchvision
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from torchvision import transforms
from tqdm import tqdm
import math
from torch.hub import download_url_to_file
import safetensors.torch as sf
import threestudio
from threestudio.utils.dge_utils import compute_epipolar_constrains, isinstance_str, make_dge_block, register_batch_idx, register_cams, register_epipolar_constrains, register_extended_attention, register_normal_attn_flag, register_pivotal
from threestudio.utils.typing import *
from threestudio.models.guidance.dge_guidance import DGEGuidance
from threestudio.utils.base import BaseObject
from threestudio.models.prompt_processors.base import PromptProcessorOutput

def modify_unet(unet):
    '''
        Change UNet.
        给UNet加上4个输入通道放I_d (32,32,4)->(32,32,8)
    '''
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    # 把unet.forward过程修改 添加了一个c_concat(input_fg编码的latent code)的内容到sample中
    # unet_original_forward = unet.forward

    # def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    #     c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    #     c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    #     new_sample = torch.cat([sample, c_concat], dim=1)
    #     kwargs['cross_attention_kwargs'] = {}

    #     return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

    # unet.forward = hooked_unet_forward
    return unet

def load_ckpt_offset(weights_path, unet):
    '''
        加载IC-light模型 是一个offset权重 直接加到原始的unet权重上.
        权重转移到GPU
        设置注意力processor

        return: device
    '''
    if not os.path.exists(weights_path):
        download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=weights_path)

    sd_offset = sf.load_file(weights_path)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)


@threestudio.register("dge-iclight")
class IClight(DGEGuidance):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        ddim_scheduler_name_or_path: str = "CompVis/stable-diffusion-v1-4"
        ip2p_name_or_path: str = "timbrooks/instruct-pix2pix"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        # diffusion_steps: int = 20
        diffusion_steps: int = 25
        use_sds: bool = False
        camera_batch_size: int = 5

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading IClight ...")
        self.weights_dtype = (
            torch.bfloat16 if self.cfg.half_precision_weights else torch.float32
        )

        # load base model
        sd15_name = "stablediffusionapi/realistic-vision-v51"
        weights_path = "threestudio/models/weights/iclight_sd15_fc.safetensors"
        tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")

        unet = modify_unet(unet)
        load_ckpt_offset(weights_path, unet)

        text_encoder = text_encoder.to(device=self.device, dtype=torch.float16)
        vae = vae.to(device=self.device, dtype=torch.bfloat16)
        unet = unet.to(device=self.device, dtype=torch.float16)

        # SDP
        unet.set_attn_processor(AttnProcessor2_0())
        vae.set_attn_processor(AttnProcessor2_0())

        # Scheduler
        dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1
        )

        t2i_pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            # image_encoder=None
        )

        # make accessible
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.scheduler = dpmpp_2m_sde_karras_scheduler
        self.t2i_pipe = t2i_pipe

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded ICLight!")
        for _, module in self.unet.named_modules():
            if isinstance_str(module, "BasicTransformerBlock"):
                make_block_fn = make_dge_block 
                module.__class__ = make_block_fn(module.__class__)
                # Something needed for older versions of diffusers
                if not hasattr(module, "use_ada_layer_norm_zero"):
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False
        register_extended_attention(self)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.unet.dtype),
            t.to(self.unet.dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.unet.dtype),
        ).sample.to(input_dtype)


    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        gaussians = None,
        cams= None,
        render=None,
        pipe=None,
        background=None,
        **kwargs,
    ):
        assert cams is not None, "cams is required for iclight"
        batch_size, H, W, _ = rgb.shape
        factor = 512 / max(W, H)
        factor = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)

        width = int((W * factor) // 64) * 64
        height = int((H * factor) // 64) * 64
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        RH, RW = height, width
        is_edit_first = kwargs.get('is_first_edit', False)
        if is_edit_first:
            # generate background image to indicate the light source
            # gradient = torch.linspace(1, 0, W, dtype=torch.float32, device=self.device)
            # image = torch.tile(gradient, (H, 1))
            # input_bg = torch.stack((image,) * 3, dim=-1)
            # rgb_BCHW = input_bg.repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2)
            # print("Using background image to indicate the light source.")

            # reading shading image as init latents
            from PIL import Image
            toTensor = transforms.ToTensor()
            shading = [toTensor(Image.open(f"edit_cache/data-dge_data-face-scene_point_cloud.ply/init_latents/{cam.uid:04d}.png")) for cam in cams]
            rgb_BCHW = torch.stack(shading, dim=0).to(self.device)

        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )

        latents = self.encode_images(rgb_BCHW_HW8)
        
        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(
            cond_rgb_BCHW,
            (RH, RW),
            mode="bilinear",
            align_corners=False,
        )
        cond_latents = self.encode_images(cond_rgb_BCHW_HW8)

        temp = torch.zeros(batch_size).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        positive_text_embeddings, negative_text_embeddings = text_embeddings.chunk(2)
        text_embeddings = torch.cat(
            [positive_text_embeddings, negative_text_embeddings, negative_text_embeddings], dim=0)  # [positive, negative, negative]
        cond_latents = torch.cat(
            [cond_latents, cond_latents, torch.zeros_like(cond_latents)], dim=0)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.max_step - 1,
            self.max_step,
            [1],
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)

        if self.cfg.use_sds:
            grad = self.compute_grad_sds(text_embeddings, latents, cond_latents, t)
            grad = torch.nan_to_num(grad)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (latents - grad).detach()
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            return {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }
        else:
            edit_latents = self.edit_latents(text_embeddings, latents, cond_latents, t, cams)
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}

    
    def image_encode(self, images):
        return self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
    
    def text_encode(self, texts):
        max_length = self.tokenizer.model_max_length
        chunk_length = self.tokenizer.model_max_length - 2
        id_start = self.tokenizer.bos_token_id
        id_end = self.tokenizer.eos_token_id
        id_pad = id_end

        def pad(x, p, i):
            return x[:i] if len(x) >= i else x + [p] * (i - len(x))

        # 处理多个文本输入
        all_tokens = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        # 对每个文本单独处理chunk和padding
        all_chunks = []
        for tokens in all_tokens:
            chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] 
                    for i in range(0, len(tokens), chunk_length)]
            chunks = [pad(ck, id_pad, max_length) for ck in chunks]
            all_chunks.extend(chunks)

        # 转换为tensor
        token_ids = torch.tensor(all_chunks).to(device=self.vae.device, dtype=torch.int64)
        encoder_hidden_states = self.text_encoder(token_ids).last_hidden_state
        encoder_hidden_states = encoder_hidden_states.to(torch.float16)

        return encoder_hidden_states
        
    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        cams= None,
    ) -> Float[Tensor, "B 4 DH DW"]:
        
        # self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        timesteps = self.scheduler.timesteps
        add_t = timesteps[1:2]
        timesteps = timesteps[timesteps<=add_t]

        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]

        camera_batch_size = self.cfg.camera_batch_size
        print("Start editing images...")

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, add_t) 

            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
            split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
            for t in timesteps:
                if t < 100:
                    self.use_normal_unet()
                else:
                    register_normal_attn_flag(self.unet, False)
                with torch.no_grad():
                    # pred noise
                    noise_pred_text = []
                    noise_pred_image = []
                    noise_pred_uncond = []
                    pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0, len(latents), camera_batch_size) 
                    register_pivotal(self.unet, True)
                    
                    key_cams = [cams[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()]
                    latent_model_input = torch.cat([latents[pivotal_idx]] * 3)
                    pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
                    pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
                    latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)

                    self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
                    register_pivotal(self.unet, False)

                    for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                        register_batch_idx(self.unet, i)
                        register_cams(self.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                        
                        epipolar_constrains = {}
                        for down_sample_factor in [1, 2, 4, 8]:
                            H = current_H // down_sample_factor
                            W = current_W // down_sample_factor
                            epipolar_constrains[H * W] = []
                            for cam in cams[b:b + camera_batch_size]:
                                cam_epipolar_constrains = []
                                for key_cam in key_cams:
                                    cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))
                                epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
                            epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)
                        register_epipolar_constrains(self.unet, epipolar_constrains)

                        batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)
                        batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
                        batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
                        batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)

                        batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                        batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                        noise_pred_text.append(batch_noise_pred_text)
                        noise_pred_image.append(batch_noise_pred_image)
                        noise_pred_uncond.append(batch_noise_pred_uncond)

                    noise_pred_text = torch.cat(noise_pred_text, dim=0)
                    noise_pred_image = torch.cat(noise_pred_image, dim=0)
                    noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

                    # perform classifier-free guidance
                    noise_pred = (
                        noise_pred_uncond
                        + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                    )

                    # get previous sample, continue loop
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    
        print("Editing finished.")
        return latents