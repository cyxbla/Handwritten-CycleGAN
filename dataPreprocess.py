from PIL import Image, ImageDraw, ImageFont
from typing import OrderedDict
from pdf2image import convert_from_path

pix = 128
s = 100
shift = 14

articleSavePath = "buffer/article_pdf"
textSavePath = "buffer/text/"
fonts = 'buffer/fonts/kaiu.ttf'
def text2image_train():
    global trainArticle
    pathSave = 'dataset/data/train/A/{:04d}.png'
    trainArticle = f'{textSavePath}/train.txt'
    
    with open(trainArticle, 'r', encoding='utf-8') as f:
        trainArticle = f.read()

    font = ImageFont.truetype(fonts, size=s, layout_engine=None)

    for i, char in enumerate(trainArticle):
        img = Image.new('RGB', (pix, pix), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((shift, shift), char, fill=(0, 0, 0), font=font) 
        img.save(pathSave.format(i))

def text2image_test():
    global testArticle, fonts
    pathSave = 'dataset/data/test/A/{:04d}.png'
    testArticle = f'{textSavePath}/test.txt'

    with open(testArticle, 'r', encoding='utf-8') as f:
        testArticle = f.read()

    font = ImageFont.truetype(fonts, size=s, layout_engine=None)

    for i, char in enumerate(testArticle):
        img = Image.new('RGB', (pix, pix), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((shift, shift), char, fill=(0, 0, 0), font=font)
        img.save(pathSave.format(i))

def splitHwrImage_train():
    # Set Train B: your own real handwritten for Train A text
    pathSave = "dataset/data/train/B/{:04d}.png"
    pages = convert_from_path(f'{articleSavePath}/data.pdf')

    fileNum = 0
    for fileNum, image in enumerate(pages):
        image.save('page{}.png'.format(fileNum), 'PNG')

    txt_len = len(trainArticle)
    i = 0
    flag = 0
    for f in range(fileNum+1):
        # open image
        im = Image.open("page{}.png".format(f))
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
            cropped_im.save(pathSave.format(i))
            # shift box
            l += diff
            r += diff
            i += 1

def splitHwrImage_test():
    # Set Test B: your own real handwritten for Test A text
    pathSave = "dataset/data/test/B/{:04d}.png"
    pages = convert_from_path(f'{articleSavePath}/Test.pdf')

    # 儲存圖像
    fileNum = 0
    for fileNum, image in enumerate(pages):
        image.save('page{}.png'.format(fileNum), 'PNG')

    txt_len = len(testArticle)
    i = 0
    flag = 0
    for f in range(fileNum+1):
        # open image
        im = Image.open("page{}.png".format(f))
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
            cropped_im.save(pathSave.format(i))
            # shift box
            l += diff
            r += diff
            i += 1
if __name__ == '__main__':
    text2image_train()
    text2image_test()
    splitHwrImage_train()
    splitHwrImage_test()