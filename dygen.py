from PIL import Image, ImageFilter 
import random, os, time
root_location = os.path.dirname(__file__)

msk = Image.open(f"{root_location}/dataset/ref/dither_256_50v.png")
msk = msk.resize((4096,4096), resample=Image.NEAREST)
msk = msk.filter(ImageFilter.GaussianBlur(radius = 1))
mskData = msk.getdata()
msk_lst = []
for i in mskData:
    if i[0] > 64:
        msk_lst.append((255,255,255))
    else:
        msk_lst.append((0,0,0))
msk.putdata(msk_lst)
#msk.save(f"{root_location}/out/{int(time.time())}_msk.png", "PNG")


img = Image.open(f"{root_location}/dataset/ref/ref_49.png")
sampleImage = img.copy().convert('RGB')
sampleImage = sampleImage.resize((4096, 4096))
sampleImageData = sampleImage.getdata()
img = img.quantize(colors=35, method=0, kmeans=1)
img = img.convert('RGB')

colors = img.getcolors()
img_data = img.copy().getdata()
imgc = img.copy()
datas = []
for color in colors:
    img = imgc.copy()
    print( f"Processing {color}")
    lst=[]
    for i in img_data:
        if i == color[1]:
            lst.append((255,255,255))
        else:
            lst.append((0,0,0))
    img.putdata(lst)
    
    if random.randint(0,3) == 0:
        img = img.resize((128,64), resample=Image.NEAREST)
    elif random.randint(0,3) == 1:
        img = img.resize((64,128), resample=Image.NEAREST)
    elif random.randint(0,3) == 2:
        img = img.resize((128,256), resample=Image.NEAREST)
    elif random.randint(0,3) == 3:
        img = img.resize((64,32), resample=Image.NEAREST)

    img = img.resize((4096,4096), resample=Image.NEAREST)

    #img = img.filter(ImageFilter.GaussianBlur(radius = 1))
    #img = img.resize((4096,4096), resample=Image.LANCZOS)
    datas.append(img.getdata())
  

lst=[]
datalen = 4096 * 4096
nofdatas = len(datas)

for k in range(datalen):
    found = False
    for dn in range(nofdatas):
        if datas[dn][k][0] > 100:
            c1 = colors[dn][1]
            c2 = sampleImageData[k]
            if msk_lst[k][0] > 64:
                if (dn % 2) == 0:
                    a1 = 1.0
                    a2 = 0.0
                else:
                    a1 = 0.0
                    a2 = 1.0
            else:
                a1 = 0.5
                a2 = 0.5

            lst.append( ( int(c1[0]*a1 + c2[0]*a2) , int(c1[1]*a1 + c2[1]*a2), int(c1[2]*a1 + c2[2]*a2)))
            found = True
            break 
    if not found:
        if msk_lst[k][0] > 64 and k < (datalen - 4096*30 - 10):
            lst.append(sampleImageData[k+4096*30])
            #lst.append((255,0,0))
        else: 
            lst.append(sampleImageData[k])
            
img.putdata(lst)
img.save(f"{root_location}/out/{int(time.time())}.png", "PNG")