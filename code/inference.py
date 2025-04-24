import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import json
import base64
import io

def model_fn(model_dir, context=None): 
    print(f"model_fn called with: model_dir={model_dir}, context={context}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return {"model": model, "processor": processor}


def input_fn(request_body, request_content_type):
    if request_content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    request = json.loads(request_body)
    conversation = request.get("inputs", {}).get("conversation", [])

    # Process the conversation, extracting and encoding images
    for message in conversation:
        if message['role'] == 'user':
            for content in message['content']:
                if content['type'] == 'image':
                    image_data = content['image'].get('data')
                    if not image_data:
                        raise ValueError("Image data is missing in the request.")
                    
                    # Decode the base64 image data
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    
                    # Save the image to a file path
                    image_path = "/tmp/image.png"  
                    image.save(image_path)
                    
                    # Update the conversation to use the image_path instead of embedding the image directly
                    content['image']['image_path'] = image_path 

    return conversation


@torch.inference_mode()
def predict_fn(input_data, model_artifacts):
    model = model_artifacts['model']
    processor = model_artifacts['processor']

    inputs = processor(
        conversation=input_data,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.9
    )
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


# Format output (simulate SageMaker response)
def output_fn(prediction, response_content_type="application/json"):
    return json.dumps({"response": prediction})
