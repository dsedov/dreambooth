import torch, os, time
import numpy as np
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from PIL import Image, ImageDraw, ImageFilter
root_location = os.path.dirname(__file__)
def dummy_checker(images, **kwargs): return images, False

model_id = "stabilityai/stable-diffusion-2-base"
#model_id = "/home/dsedov/Dev/dreambooth/models/dystorefront"
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

seed = 134334552
generator = torch.Generator(device="cuda")
generator = generator.manual_seed(seed)


txt2img = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, revision="fp16")
img2img = StableDiffusionImg2ImgPipeline(**txt2img.components)
inpaint = StableDiffusionInpaintPipeline(**txt2img.components)
def callback(step, timestep, latents):
    #print(latents.shape)
    x=0
def upscale(image, mask, prompt, scale_factor, tile_size_w, tile_size_h, tile_step, strength=0.5, gf=7.5, steps=50, artist="" ):
    image = image.resize(( int(image.size[0] * scale_factor),int(image.size[1] * scale_factor)), resample=Image.Resampling.LANCZOS)
    maskL = mask.convert('L')
    for y in range(0, image.size[1] // tile_step + 2):
        for x in range(0, image.size[0] // tile_step + 2):
            print(f"tile {x}:{y}")
            sendImage = image.crop((x * tile_step - tile_size_w//2, y * tile_step - tile_size_h//2, x * tile_step + tile_size_w//2, y * tile_step + tile_size_h//2))
            pipe = img2img.to("cuda")
            pipe.censored = False
            #pipe.safety_checker = dummy_checker
            if artist != "": final_prompt=prompt.replace('ARTIST', artist)
            else: final_prompt = prompt
            img = pipe(prompt=final_prompt, censored=False, init_image=sendImage, mask=maskL, strength=strength, generator=generator, callback=callback, guidance_scale=gf, num_inference_steps=steps).images[0] 
            img_comp = Image.composite(sendImage, img, maskL)
            Image.Image.paste(image, img_comp, (x * tile_step - tile_size_w//2, y * tile_step - tile_size_h//2))
            
    return image

# General properties
project_name = "test"

# Prompt
initial_prompt = "beautiful geometric fluid antropomorphic giant castle underground tonnels maze in the middle of nowhere, liquid quicksilver, splashes of colorful paint, rainbow background, volume fog, volume light"
upscale_prompt = initial_prompt
upscale_prompt = "oilpainting, texture, detailed, brushwork"
upscale_prompt = "dripping paint, splashes, colorful, oil painting, texture, detailed, brushwork"
upscale_prompt = "generative design, square pattern, geometric lines"
upscale_prompt = "letters, fonts, pile of words and letters, by Tatsuro Kiuchi"
upscale_prompt = "letters, fonts, pile of words and letters, by Sonia Delaunay"
upscale_prompt = "vertical rainbow lines, geometric, by Sonia Delaunay"
upscale_prompt = "downtown los angeles streets, early morning, by Richard Diebenkorn"
upscale_prompt = "abstract forms by Paula Modersohn-Becker"
upscale_prompt = "dripping paint oil, abstract forms by Paula Modersohn-Becker" # **

upscale_prompt = "patchwork of colors, geometric blobs of oil paint with rounded corners and a bit of noise, dots dripping down, by Paula Modersohn-Becker, by " # **

upscale_prompt = "patchwork of colors, geometric blobs of oil paint with rounded corners and a bit of noise, dots dripping down, by ARTIST, by " # **

upscale_prompt = "minimalist graphic painting by ARTIST of geometric blobs with rounded corners, fluid forms, vertical and horizontal by ARTIST" # **

upscale_prompt = "minimalist graphic painting by ARTIST of geometric blobs with rounded corners,fluid forms, vertical and horizontal by ARTIST, oil paint texture, palette" # **

#upscale_prompt = "a picture of a castle on top of a mountain, behance contest winner, abstract illusionism, magical portal opened, crumbling, slight color bleed, drip"
#upscale_prompt = "a picture of a castle on top of a mountain, behance contest winner, abstract illusionism, magical portal opened, crumbling, slight color bleed, drip, flat colour, hollow, gateway, highly detailed saturated"

#upscale_prompt = "dripping paint oil, abstract forms by Robert Antoine Pinchon"

#upscale_prompt = "rounded minimalist abstract art, dan mcpharlin, 1968 science fiction tarot card, by Sophie Taeuber-Arp"
#upscale_prompt = "dripping paint oil, abstract forms by Umberto Boccioni" # **

# Henry Raeburn
# Konstantin Makovsky

#upscale_prompt = "painting of gunmetal steel plates, by Paula Modersohn-Becker"
#upscale_prompt = "crumbled recycled paper, cardboard, abstract forms by Paula Modersohn-Becker"
#upscale_prompt = "bookshelf library, by Paula Modersohn-Becker"
#upscale_prompt = "downtown los angeles streets, early morning, by Odilon Redon"
#upscale_prompt = "downtown los angeles streets, early morning, by Natalia Goncharova"
#upscale_prompt = "downtown los angeles streets, early morning, by Mordecai Ardon"
#upscale_prompt = "downtown los angeles streets, early morning, by Jean Delville"
#upscale_prompt = "downtown los angeles streets, early morning, by Frank Auerbach"
#upscale_prompt = "downtown los angeles streets, early morning, by Andre Derain"
artists = [
    # "Will Barnet",
    #"Jean Arp", *
    #"Tomma Abts",
    #"Craigie Aitchison",
    # "Louis Anquetin",
    #"Ben Arson",
    #"George Ault",
    "Milton Avery",
    #"George Birrell",
    #"Yaacov Agam",
    #"Paula Modersohn-Becker",
    #"Etel Adnan",
]
initial_file = "ref_45.png"

# Tech settings
scale_factor = 4 
tile_size_w = 512
tile_size_h = 512
tile_mask_errode = 64
tile_mask_blur = 16
tile_step = 512-128-32

# Generate initial image
if initial_file == "":
    pipe = txt2img.to("cuda")
    pipe.safety_checker = dummy_checker
    image = pipe(initial_prompt).images[0]  
    pilImage = image
else:
    pilImage = Image.open(f"{root_location}/dataset/ref/{initial_file}")
    pilImage = pilImage.convert('RGB')

# Generate mask
pilMask = Image.new(mode='RGB', size= (tile_size_w,tile_size_h), color=(255,255,255))
draw_handle = ImageDraw.Draw(pilMask)
draw_handle.rectangle([(tile_mask_errode,tile_mask_errode),(tile_size_w - tile_mask_errode,tile_size_h - tile_mask_errode)], fill = (0,0,0))
pilMask = pilMask.filter(ImageFilter.GaussianBlur(tile_mask_blur))
pilImageCopy = pilImage.copy()

scale_factors = [2]
guidance_factors = [10]
images = ['ref_32.png', 'ref11.png', 'ref20.png', 'ref31.png', 'ref29.png' ]
steps = [100,100]
strengths = [0.45, 0.5, 0.55, 0.6, 0.7]
for image in images:
    for artist in artists:
        for step in steps:
        
            pilImage = Image.open(f"{root_location}/dataset/ref/{image}")
            pilImage = pilImage.convert('RGB')
            pilImageCopy = pilImage.copy()
            for guidance_factor in guidance_factors:
                for scale_factor in scale_factors:
                    for i in strengths:
                        print(f"\n\n\n\n range {i}")
                        pilImage = upscale(pilImageCopy, pilMask, upscale_prompt, scale_factor, tile_size_w, tile_size_h, tile_step, strength=i, gf=guidance_factor, steps=step, artist=artist)
                        pilImage.save(f"{root_location}/out/{int(time.time())}_{image}_{artist}_x{scale_factor}_s{i:.2f}_gf{guidance_factor:.1f}_steps{step}.png", "PNG")

#scale_factor = 3
#pilImage = upscale(pilImage, pilMask, upscale_prompt, scale_factor, tile_size_w, tile_size_h, tile_step, strength=0.7)
#pilImage.save(f"{root_location}/out/{int(time.time())}.png", "PNG")
