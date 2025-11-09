import copy
import gc
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import base64
from io import BytesIO
import os

class QwenVQAModel:

    def __init__(self, model_path: str|None=None, device=None) -> None:
        if model_path is None:
            model_path = "/mnt/16T/yejiangnan/.cache/hf_models/Qwen2.5-VL-7B-Instruct"
            # model_path = "/data/vjuicefs_ai_camera_ql/11184175/models/Qwen2.5-VL-7B-Instruct"
        self.model_path = model_path
            
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
            
        self.model = None
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        assets_folder = Path(__file__).parents[2] / "test_assets"
        self.messages = [
            # system 指令
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": 
"""You are an AI vision assistant responsible for analyzing input images and generating 3 appropriate text prompts for image relighting tasks. The generated text prompts should be accurate, reflect the image content, and include details about lighting, illumination, tone, or shadow effects. 
Here are some prompt exsample:
- "beautiful woman, detailed face, neon, Wong Kar-wai, warm"
- "handsome boy, detailed face, neon light, city"
- "toy, detailed face, shadow from window"
- "dog, magic lit, sci-fi RGB glowing, studio lighting"
Here are some useful expression:
"sunshine from window", "neon light, city", "sunset over sea", "golden time", "sci-fi RGB glowing, cyberpunk", "natural lighting", "warm atmosphere, at home, bedroom", "magic lit", "evil, gothic, Yharnam", "light and shadow", "shadow from window", "soft studio lighting", "home atmosphere, cozy bedroom illumination", "neon, Wong Kar-wai, warm"
Generate 3 prompts for each response, always answer strictly in the format:
["Prompt 1", "Prompt 2", "Prompt 3"]
Do not add any other words or explanations."""
                    }
                ]
            },
            # Few-shot case 1
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{Path(assets_folder) / 'vqa' / 'face.jpg'}",
                    },
                    {"type": "text", "text": "Give me 3 text prompts to relight this image."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[\"detailed face, sunshine, indoor, warm atmosphere\", \"handsome boy, detailed face, neon light, indoor\", \"detailed clear face, indoor, cool tunes, stage lighting, blue spotlight\"]"}
                ]
            },
            # Few-shot case 2
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{Path(assets_folder) / 'vqa' / 'farm.jpg'}",
                    },
                    {"type": "text", "text": "Give me 3 text prompts to relight this image."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[\"broad farm, outdoor, scorching sun, summer, hot atmosphere\", \"open grassland, outdoor, overcast sky, gloomy environment\", \"a lush tree, outdoor, starry night, silver moon light, sanctity\"]"}
                ]
            },
            # Few-shot case 3
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{Path(assets_folder) / 'vqa' / 'office.png'}",
                    },
                    {"type": "text", "text": "Give me 3 text prompts to relight this image."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[\"office, indoor, sunshine from window, warm atmosphere\", \"office, indoor, dim candle light, dark\", \"office, indoor, evil, gothic, Yharnam, darkgreen tone\"]"}
                ]
            },
        ]

    def load_model(self):
        if self.model is None:
            print("Loading Qwen2.5-vl model...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16, 
                device_map="cuda",
                # attn_implementation="flash_attention_2"
                attn_implementation="eager"
            )
            
    def unload_model(self):
        if self.model is not None:
            print("Unloading Qwen2.5-vl model...")
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            
    # 进入 with 时自动加载模型
    def __enter__(self):
        self.load_model()
        return self
    
    # 退出 with 时自动卸载模型并清空显存
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()
        return False  # 如果 with 块中抛出异常，不会被吞掉
    
    def inference(self, image: str | Image.Image, prompt: str):
        """
            image: (str|Image)
                input image for VQA
                if str, assume it as image_path
                if Image, encode it as base64
            prompt: str
                input editing prompt
        """
        if isinstance(image, str):
            image_content = f"file://{os.path.abspath(image)}"
        elif isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            image_content = f"data:image;base64,{img_base64}"
            
        if self.model is None:
            self.load_model()
            
        # step 1. construct full message
        # Inference case
        addition_query = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_content,
                },
                {"type": "text", "text": f"This is an image which will be edited according to the text prompt: \"{prompt}\". Tell me what light source position is relative to which objects in the edited image."},
            ],
        }
        messages = copy.deepcopy(self.messages)
        messages.append(addition_query)
        
        # step 2. extract language and vision prompt
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages) # type: ignore
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # step 3. inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128) # type: ignore
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode( # type: ignore
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]