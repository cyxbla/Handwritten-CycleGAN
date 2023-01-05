import random
import subprocess
import math
import pywebio
import time
from pywebio.input import * 
from pywebio.output import * 
from pywebio.session import run_js
from pathlib import Path
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
from random import seed
seed(random.randint(0,99))

input_text_path = 'buffer/article_text/input.txt'
fonts = "buffer/fonts/kaiu.ttf"
text_img_pathsaveA = "datasets/predict/test/A/"
text_img_pathsaveB = "datasets/predict/test/B/"
hwr_img_pathsave = "output/A"
log_buffer = "buffer/log.log"
class Paths(Enum):
    YenXuan = Path("output/path/yenxuan/yenxuan_199_netG_A2B.pth")

    NienEn = "output/path/chris/chris_299_netG_A2B.pth"

    YuWen = "output/path/winwu/winwu_199_netG_A2B.pth"

    YuXuan = "output/path/yuxuan/yuxuan_no_flip_159_netG_A2B.pth"


article_len = 0
punctuation = {}
newline = 0
def init():
    global article_len
    subprocess.run(["rm", "-rf", f"{text_img_pathsaveA}"])
    subprocess.run(["rm", "-rf", f"{text_img_pathsaveB}"])
    subprocess.run(["rm", "-rf", f"{hwr_img_pathsave}"])
    subprocess.run(["mkdir", f"{text_img_pathsaveA}"])
    # subprocess.run(["mkdir", f"{text_img_pathsaveB}"])
    subprocess.run(["mkdir", f"{hwr_img_pathsave}"])
    article_len = 0
    getWord()

def show_result():
    img = open('result.png', 'rb').read()  
    pywebio.output.put_image(img) 
    pywebio.output.put_grid([
            [None], 
        ], cell_width='125px', cell_height='30px') ## padding
    
    pywebio.output.put_grid([
        [pywebio.output.put_button("Download", onclick=lambda: pywebio.session.download('result.png', img), color='success', outline=True), 
        pywebio.output.put_button("Try Again", onclick=lambda: run_js('window.location.reload()'), color='success', outline=True)],
    ], cell_width='125px', cell_height='100px')

def create_img():
    global article_len
    pywebio.output.put_text("create image...")
    pywebio.output.put_processbar('create_img')
    
    images = []
    for i in range(article_len):
        tmp = Image.open("output/A/{:04d}.png".format(i))
        images.append(tmp.resize((64, 64)))
    max_width = 832
    img_width = 64
    img_height = 64
    
    col = max_width // img_width
    col = 15        # FIX
    row = math.ceil(article_len / float(col))+newline
    
    result = Image.open(f"background/{background}.png")
    result = result.resize((max_width, row*64))
    x = 0
    y = 0       
    i = 0
    
    for image in images:
        if i in punctuation:
            if punctuation[i] == 2:
                y += img_height
                x = 0
                i += 1
                continue
        box = (3+random.randint(1,10), 0, 61-random.randint(1,10), 64) # (l, t, r, b)
        image = image.crop(box)
        im = Image.new("RGB", (random.randint(50, 60), 64), (255,255,255))
        im.paste(image, (random.randint(0,5), random.randint(0,5)))
        if x + im.width > max_width:
            x = 0
            y += img_height
        mask = get_mask(im)
        result.paste(im, (x, y), mask)
        i += 1
        x += im.width 
        pywebio.output.set_processbar('create_img', i / article_len)
        time.sleep(0.1)  
        
    result.save("result.png")
    show_result()


def generate_handwritten():
    pywebio.output.put_text("generate handwritten...")
    pywebio.output.put_processbar('generate_handwritten')
    subprocess.run(["cp", "-rf", f"{text_img_pathsaveA}", f"{text_img_pathsaveB}"])
    pywebio.output.set_processbar('generate_handwritten', 5 / 10)
    time.sleep(0.1)
    subprocess.run(["python3", "predict.py", "--cuda", "--dataroot", "datasets/predict", "--generator_A2B", f"{Paths[selected].value}"])
    pywebio.output.set_processbar('generate_handwritten', 1)
    time.sleep(0.1)
    create_img()

def generate_text_image():
    global article_len, newline
    pix = 128
    s = 100
    shift = 14

    with open(input_text_path, "r") as f:
        user_input = f.read()
    for i, ch in enumerate(user_input):
        if(ch == '，' or ch == '。' or ch == '：' or ch == '？' or ch == '「' or ch == '」'):
            punctuation[i] = 1
        elif ch == '\n':
            newline += 1
            punctuation[i] = 2
        article_len += 1
        image = Image.new("RGB", (pix, pix), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(fonts, size=s, layout_engine=None)

        draw.text((shift, shift), ch, fill=(0, 0, 0), font=font)

        image.save(text_img_pathsaveA + "{:04n}".format(i) +".png")
    generate_handwritten()

def getWord():
    global selected, background
    arcle = pywebio.input.textarea('Auto Memory Dolls', rows=6, maxlength=500, minlength=1)
    pywebio.output.put_text(arcle)
    selected = pywebio.input.select("Select Fonts:", ["YenXuan", "NienEn", "YuWen", "YuXuan"])
    background = pywebio.input.select("Select Background:", ["defult","rainbow", "marble", "wood"])
    with open(input_text_path, 'w') as f:
        f.write(arcle)   
    
    print(selected, background)
    generate_text_image()

def get_mask(pic):
    width,height = pic.size
    pixels = pic.load()
    mask = Image.new("L",(width,height),255)
    mask_p = mask.load()
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            gray = (r+g+b) // 3
            if gray > 122:
                mask_p[x,y] = 0
            else:
                mask_p[x,y] = 255
    return mask
if __name__ == '__main__':
    pywebio.config(title="MLFianl Team 10 Demo")
    pywebio.platform.tornado.start_server(init, port=8080, log_file="buffer/file.log", debug=True, host="localhost")