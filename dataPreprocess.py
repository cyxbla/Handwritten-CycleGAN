from PIL import Image, ImageDraw, ImageFont
from typing import OrderedDict
from pdf2image import convert_from_path
import os
pix = 128
s = 100
shift = 14

dir = "test"

trainArticleText = "buffer/article_text/yuxuan.txt"
testArticleText = "buffer/article_text/test.txt"
trainArticlePDF = "buffer/article_pdf/yuxuan.pdf" 
testArticlePDF = "buffer/article_pdf/test.pdf" 

trainText2ImageSave = f"datasets/{dir}/train/A"
testText2ImageSave = f"datasets/{dir}/test/A"
trainHwrImageSave = f"datasets/{dir}/train/B"
testHwrImageSave = f"datasets/{dir}/test/B"

fonts = 'buffer/fonts/kaiu.ttf'
pdfPage = "buffer/pages"

if not os.path.exists(f'{trainText2ImageSave}'):
    os.makedirs(f'{trainText2ImageSave}')
if not os.path.exists(f'{testText2ImageSave}'):
    os.makedirs(f'{testText2ImageSave}')
if not os.path.exists(f'{trainHwrImageSave}'):
    os.makedirs(f'{trainHwrImageSave}')
if not os.path.exists(f'{testHwrImageSave}'):
    os.makedirs(f'{testHwrImageSave}')

def text2image_train():
    global trainArticleText, trainText2ImageSave
    pathSave = '{}/{:04d}.png'
    
    with open(trainArticleText, 'r', encoding='utf-8') as f:
        trainArticleText = f.read()

    font = ImageFont.truetype(fonts, size=s, layout_engine=None)

    for i, char in enumerate(trainArticleText):
        img = Image.new('RGB', (pix, pix), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((shift, shift), char, fill=(0, 0, 0), font=font) 
        img.save(pathSave.format(trainText2ImageSave, i))

def text2image_test():
    global testArticleText, fonts, testText2ImageSave
    pathSave = '{}/{:04d}.png'

    with open(testArticleText, 'r', encoding='utf-8') as f:
        testArticleText = f.read()

    font = ImageFont.truetype(fonts, size=s, layout_engine=None)

    for i, char in enumerate(testArticleText):
        img = Image.new('RGB', (pix, pix), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((shift, shift), char, fill=(0, 0, 0), font=font)
        img.save(pathSave.format(testText2ImageSave, i))

def splitHwrImage_train():
    # Set Train B: your own real handwritten for Train A text
    pathSave = "{}/{:04d}.png"
    pages = convert_from_path(f'{trainArticlePDF}')

    fileNum = 0
    for fileNum, image in enumerate(pages):
        image.save(f'{pdfPage}/page{fileNum}.png', 'PNG')

    txt_len = len(trainArticleText)
    i = 0
    flag = 0
    for f in range(fileNum+1):
        # open image
        im = Image.open(f'{pdfPage}/page{f}.png')
        w, h = im.size

        # (l, t, r, b) left, top, right, bottom
        (l, t, r, b) = (100, 0, 400, 300)
        diff = 300

        while True:
            # if some texts are missing or redundant, uncomment below code
            # if(i == 69 and flag == 0):
            #   flag = 1
            #   l += diff
            #   r += diff
            #   continue
            if(i >= txt_len):
                break 
            
            # the crop range
            if(r > w or l > w):
                t += diff
                b += diff
                l = 100
                r = 400
            if(b > h or t > h):
                break
            box = (l, t, r, b)

            # using crop() method
            cropped_im = im.crop(box)
            cropped_im = cropped_im.resize((pix, pix))
            # convert to grayscale
            # cropped_im = cropped_im.convert('1')
            # save image
            cropped_im.save(pathSave.format(trainHwrImageSave, i))
            # shift box
            l += diff
            r += diff
            i += 1

def splitHwrImage_test():
    # Set Test B: your own real handwritten for Test A text
    pathSave = "{}/{:04d}.png"
    pages = convert_from_path(f'{testArticlePDF}')

    # 儲存圖像
    fileNum = 0
    for fileNum, image in enumerate(pages):
        image.save(f'{pdfPage}/page{fileNum}', 'PNG')

    txt_len = len(testArticleText)
    i = 0
    flag = 0
    for f in range(fileNum+1):
        # open image
        im = Image.open(f'{pdfPage}/page{f}.png')
        w, h = im.size

        # (l, t, r, b) left, top, right, bottom
        (l, t, r, b) = (100, 0, 400, 300)
        diff = 300

        while True:
            # if some texts are missing or redundant, uncomment below code
            # if(i == 69 and flag == 0):
            #   flag = 1
            #   l += diff
            #   r += diff
            #   continue
            if(i >= txt_len):
                break 
            
            # the crop range
            if(r > w or l > w):
                t += diff
                b += diff
                l = 100
                r = 400
            if(b > h or t > h):
                break
            box = (l, t, r, b)

            # using crop() method
            cropped_im = im.crop(box)
            cropped_im = cropped_im.resize((pix, pix))
            # convert to grayscale
            # cropped_im = cropped_im.convert('1')
            # save image
            cropped_im.save(pathSave.format(testHwrImageSave, i))
            # shift box
            l += diff
            r += diff
            i += 1
if __name__ == '__main__':
    text2image_train()
    splitHwrImage_train()
    # text2image_test()
    # splitHwrImage_test()