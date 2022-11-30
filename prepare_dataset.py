import os, re
from PIL import Image

from transformers import GPT2TokenizerFast, ViTFeatureExtractor, VisionEncoderDecoderModel

dataset_location = "/home/dsedov/Dev/dreambooth/dataset/storefront"

# load a fine-tuned image captioning model and corresponding tokenizer and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

i = 0
dataset_files = [f for f in os.listdir(dataset_location) if os.path.isfile(os.path.join(dataset_location, f))]
with open(f"{dataset_location}/metadata.jsonl", "w", encoding='utf-8') as f:
    for file_name in dataset_files:
        if file_name.endswith('.jsonl') : continue
        if file_name.startswith('.') : continue
        
        url = f"{dataset_location}/{file_name}"
        img = Image.open(url)
        if img.size[0] > img.size[1]:
            img = img.crop(( (img.size[0] - img.size[1])//2, 
                            0, 
                            img.size[1] + (img.size[0] - img.size[1])//2, 
                            img.size[1]
                            ))
        else:
            img = img.crop(( 0,                             
                            (img.size[1] - img.size[0])//2,
                            img.size[0],
                            img.size[0] + (img.size[1] - img.size[0])//2
                            ))
        img = img.resize((512,512), resample=Image.LANCZOS)
        img.save(f"{dataset_location}/item_{i}.png", 'PNG')

        pixel_values = feature_extractor(img, return_tensors="pt").pixel_values

        # autoregressively generate caption (uses greedy decoding by default)
        generated_ids = model.generate(pixel_values, max_length=30)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_text = re.sub(r'[^a-zA-Z0-9\s]', '', generated_text)
        generated_text = generated_text.strip() + ", painting by dymokomi"
        print(generated_text)
        f.write('{ "file_name" : "')
        f.write(f"item_{i}.png")
        f.write('", "text" : "' + generated_text + '"}\n')

        i += 1
