from tkinter import *
from tkinter import PhotoImage
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
import subprocess
import os
import math

PATH = "C:\\Users\Hairy\Desktop\ML"
filepath = os.path.join(PATH, "text.txt")
root = tk.Tk()
root.geometry("1000x700")
root.title("ML Team10 DEMO")
article_len = 0
def init():
    global article_len
    subprocess.run(["wsl", "rm", "-rf", "/mnt/c/Users/Hairy/Desktop/ML/datasets/predict/test/B/"])
    subprocess.run(["wsl", "rm", "-rf", "/mnt/c/Users/Hairy/Desktop/ML/datasets/predict/test/A/"])
    subprocess.run(["wsl", "rm", "-rf", "/mnt/c/Users/Hairy/Desktop/ML/output/A/*"])
    subprocess.run(["wsl", "mkdir", "/mnt/c/Users/Hairy/Desktop/ML/datasets/predict/test/B/"])
    subprocess.run(["wsl", "mkdir", "/mnt/c/Users/Hairy/Desktop/ML/datasets/predict/test/A/"])
    article_len = 0

def show_result():
    window = tk.Toplevel()
    window.title("Handwritten Result")
    window.geometry("832x500")

    main_frame = Frame(window)
    main_frame.pack(fill=BOTH, expand=1)

    canvas = Canvas(main_frame)
    canvas.pack(side=LEFT, fill=BOTH, expand=1)

    scrollbar = Scrollbar(window, orient=VERTICAL, command=canvas.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion= canvas.bbox("all")))
    img = Image.open("result.png")
    photo = ImageTk.PhotoImage(img)
    
    second_frame = Frame(canvas)
    label_img = Label(second_frame, image=photo)
    label_img.pack()
    canvas.create_window((0, 0), window=second_frame, anchor="nw")
    window.mainloop()

def create_img():
    input.delete(1.0, END)

    images = []
    for i in range(article_len-1):
        tmp = Image.open("output/A/{:04d}.png".format(i))
        images.append(tmp.resize((64, 64)))
    max_width = 832
    img_height = 64
    col = max_width // article_len
    row = math.ceil(article_len / float(col))
    print(row)
    result = Image.new("RGB", (max_width, row*64), (255,255,255))
    x = 0
    y = 0
    for image in images:
        if x + image.width > max_width:
            x = 0
            y += img_height
        result.paste(image, (x, y))
        x += image.width   
        print(result.size) 
    result.save("result.png")
    show_result()


def generate_handwritten():
    subprocess.run(["wsl", "cp", "-rf", "/mnt/c/Users/Hairy/Desktop/ML/datasets/predict/test/B/*", "/mnt/c/Users/Hairy/Desktop/ML/datasets/predict/test/A"])
    subprocess.run(["wsl", "python3", "predict.py", "--cuda", "--dataroot", "datasets/predict", "--generator_A2B", "199_netG_B2A.pth"])
    create_img()

def generate_text_image():
    global article_len
    pix = 128
    s = 100
    shift = 14

    fonts = f"{PATH}/kaiu.ttf"
    pathSave = f"{PATH}/datasets/predict/test/B/"

    with open(filepath, "r") as f:
        user_input = f.read()
    for i, ch in enumerate(user_input):
        article_len += 1
        image = Image.new("RGB", (pix, pix), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(fonts, size=s, layout_engine=None)

        draw.text((shift, shift), ch, fill=(0, 0, 0), font=font)

        image.save(pathSave + "{:04n}".format(i) +".png")
    generate_handwritten()

def save_text():
    init()
    text = input.get(1.0, END)
    with open(filepath, "w") as f:
        f.write(text)
    generate_text_image()

# text box
input = Text(root, width=80, height=30, font=("kaiu", 16))
input.pack(pady= 10)

# button 
button_frame = Frame(root)
button_frame.pack()
submit_button = Button(button_frame, text="Submit", command=save_text)
submit_button.grid(row=0, column=1, padx=20)

root.mainloop()
