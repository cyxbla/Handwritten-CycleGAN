from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from random import seed
from random import gauss
import random
import subprocess
import os
import math
import pywebio
seed(44)
# PATH = "C:\\Users\Hairy\Desktop\ML"
base = "/mnt/c/home/hchiang/test/Handwritten-CycleGAN"
input_text_path = 'buffer/input.txt'
fonts = "buffer/kaiu.ttf"
text_img_pathsave = "datasets/predict/test/A/"
hwr_img_pathsave = "output/A"
log_buffer = "buffer/log.log"

article_len = 0
punctuation = {}
def init():
    global article_len
    subprocess.run(["rm", "-rf", f"{text_img_pathsave}"])
    subprocess.run(["rm", "-rf", f"{hwr_img_pathsave}"])
    subprocess.run(["mkdir", f"{hwr_img_pathsave}"])
    subprocess.run(["mkdir", f"{text_img_pathsave}"])
    article_len = 0
    getWord()

def show_result():
    img = open('result.png', 'rb').read()  
    pywebio.output.put_image(img) 
def create_img():
    global article_len
    images = []
    for i in range(article_len-1):
        tmp = Image.open("output/A/{:04d}.png".format(i))
        images.append(tmp.resize((64, 64)))
    max_width = 832
    img_width = 64
    img_height = 64
    
    col = max_width // img_width
    col = 15        # FIX
    row = math.ceil(article_len / float(col))
    result = Image.new("RGB", (max_width, row*64), (255,255,255))
    x = 0
    y = 0       
    i = 0
    for image in images:
        (l, t, r, b) = (3+random.randint(1,10), 0, 61-random.randint(1,10), 64)
        if i not in punctuation:
            (l, t, r, b) = (3+random.randint(1,10), 0, 61-random.randint(1,10), 64)
        elif punctuation[i] == ['\n']:
            print("new line")
            y += img_height
            x = 0
        else:
            (l, t, r, b) = (4, 4, 4, 4)            
        box = (l, t, r, b)
        image = image.crop(box)
        im = Image.new("RGB", (random.randint(50, 60), 64), (255,255,255))
        im.paste(image, (random.randint(0,5), random.randint(0,5)))
        # im.save("{:04d}.png".format(i))
        if x + im.width > max_width:
            x = 0
            y += img_height
        
        result.paste(im, (x, y))
        i += 1
        x += im.width   
        
    result.save("result.png")
    show_result()


def generate_handwritten():
    # subprocess.run(["cp", "-f", f"{base}/datasets/predict/test/B/*", f"{base}/datasets/predict/test/A"])
    subprocess.run(["python3", "predict.py", "--cuda", "--dataroot", "datasets/predict", "--generator_A2B", "output/199_netG_B2A.pth"])
    create_img()

def generate_text_image():
    global article_len
    pix = 128
    s = 100
    shift = 14

    with open(input_text_path, "r") as f:
        user_input = f.read()
    for i, ch in enumerate(user_input):
        article_len += 1
        image = Image.new("RGB", (pix, pix), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(fonts, size=s, layout_engine=None)

        draw.text((shift, shift), ch, fill=(0, 0, 0), font=font)

        image.save(text_img_pathsave + "{:04n}".format(i) +".png")
    generate_handwritten()

def getWord():
    # init()
    arcle = pywebio.input.textarea('想要轉換的文章', rows=6)
    # pywebio.input.actions('actions', [
    #     {'label': 'Run', 'value': 'save'},
    #     {'label': 'Reset', 'type': 'reset', 'color': 'warning'}
    # ], help_text='actions')
    # runCode()
    pywebio.output.put_text(arcle)
    with open(input_text_path, 'w') as f:
        f.write(arcle)
    generate_text_image()
    pywebio.output.put_button("重新製作", onclick=lambda: init(), color='success', outline=True)
    

if __name__ == '__main__':
    pywebio.platform.tornado.start_server(init, port=8080, log_file="buffer/log.log", auto_reload=True)