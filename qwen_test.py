from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    # "/data/vjuicefs_ai_camera_vgroup_ql/11184175/models/Qwen2.5-VL-7B-Instruct", 
    "/homeC/Public/yjn_data/hf_models/Qwen2.5-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda",
    attn_implementation="flash_attention_2"
)

# processor = AutoProcessor.from_pretrained("/data/vjuicefs_ai_camera_vgroup_ql/11184175/models/Qwen2.5-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("/homeC/Public/yjn_data/hf_models/Qwen2.5-VL-7B-Instruct")
# input_image_path = "/data/juicefs_sharing_data/11184175/work/VPP-LLaVA/test_asset/front.png"
input_image_path = "/homeB/Public/yejiangnan/work/dge/edit_cache/data-dge_data-face-scene_point_cloud.ply/origin_render/0020.png"
edit_prompt = "detailed face, sunshine, indoor, warm atmosphere, light from left side"
print(f"Input image: {input_image_path}")
print(f"Editing prompt: {edit_prompt}")

messages = [
    # system 指令
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """You are an AI visual assistant. You analyze images and the editing text prompts 
                    to determine that what light source position is relative to which objects in the edited image.
                    Always answer strictly in the format:
                    {"Position": #POSITION#, "Object": #OBJECT#}
                    Where:
                    - #POSITION# is an single word among "left", "right", "top", "bottom", "front" and "back".
                    - #OBJECT# is an object mentioned in the text prompt or the image, and the text prompt takes priority over the image.
                    This means that the light is on the #POSITION# of #OBJECT#. Do not add any other words or explanations."""
            }
        ]
    },
    # Few-shot case 1
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": "file:///data/juicefs_sharing_data/11184175/work/VPP-LLaVA/test_asset/front.png",
                "image": "file:///homeB/Public/yejiangnan/work/dge/edit_cache/data-dge_data-face-scene_point_cloud.ply/origin_render/0000.png",
            },
            {"type": "text", "text": "This is an image which will be edited according to the text prompt: \"cyberpunk, neon light, indoor\". Tell me what light source position is relative to which objects in the edited image."}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "{\"Position\": \"right\", \"Object\": \"face\"}"}
        ]
    },
    # Few-shot case 2
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "This is an unknown image which will be edited according to the text prompt: \"The Buddha statue in meditation emits light from behind\". Tell me what light source position is relative to which objects in the edited image."}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "{\"Position\": \"back\", \"Object\": \"the Buddha statue\"}"}
        ]
    },
    # Inference case
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"file://{os.path.abspath(input_image_path)}",
            },
            {"type": "text", "text": f"This is an image which will be edited according to the text prompt: \"{edit_prompt}\". Tell me what light source position is relative to which objects in the edited image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"Answer: {output_text[0]}")