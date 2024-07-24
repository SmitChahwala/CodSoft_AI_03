# IMAGE CAPTIONING

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch 
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_caption(image_paths):
    images = []
    for image_path in image_paths:
        i_iamge = Image.open(image_path)
        if i_iamge.mode != "RGB":
            i_iamge = i_iamge.convert(mode="RGB")
        images.append(i_iamge)
        
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    output_ids = model.generate(pixel_values, **gen_kwargs)
    
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    print("Final Caption is: ", preds)
    return preds

predict_caption(['animals.jpeg'])
    