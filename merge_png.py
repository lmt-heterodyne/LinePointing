import os
import sys
import traceback
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

def merge_png(image_files, newfile, direction='v', resize=True):
    try:
        print ('merge_png', image_files, newfile)
        images = list(map(Image.open, image_files))
        widths, heights = zip(*(i.size for i in images))

        if direction == 'v':
            max_width = max(widths)
            total_height = 0
            for im in images:
                if resize:
                    total_height = total_height + im.size[1]*max_width/im.size[0]
                else:
                    total_height = total_height + im.size[1]

            new_im = Image.new('RGB', (int(max_width), int(total_height)))

            x_offset = 0
            y_offset = 0
            for im in images:
                if resize:
                    try:
                        new_im.paste(im.resize(
                            (int(max_width), int(im.size[1]*max_width/im.size[0])), Image.ANTIALIAS), (int(x_offset), int(y_offset)))
                    except:
                        new_im.paste(im.resize(
                            (int(max_width), int(im.size[1]*max_width/im.size[0])), Image.LANCZOS), (int(x_offset), int(y_offset)))
                    y_offset += im.size[1]*max_width/im.size[0]
                else:
                    new_im.paste(im, (int(x_offset), int(y_offset)))
                    y_offset += im.size[1]
        else:
            max_height = max(heights)
            total_width = 0
            for im in images:
                if resize:
                    total_width = total_width + im.size[0]*max_height/im.size[1]
                else:
                    total_width = total_width + im.size[0]

            new_im = Image.new('RGB', (int(total_width), int(max_height)))

            x_offset = 0
            y_offset = 0
            for im in images:
                if resize:
                    try:
                        new_im.paste(im.resize(
                            (int(im.size[0]*max_height/im.size[1]), int(max_height))), (int(x_offset), int(y_offset)))
                    except:
                        new_im.paste(im.resize(
                            (int(im.size[0]*max_height/im.size[1]), int(max_height))), (int(x_offset), int(y_offset)))

                    x_offset += im.size[0]*max_height/im.size[1]
                else:
                    new_im.paste(im, (int(x_offset), int(y_offset)))
                    x_offset += im.size[0]

        new_im.save(newfile)
        #os.system('rm -f %s'%zip(image_files))

    except Exception as e:
        print (e)
        traceback.print_exc()
        pass


    
def watermark_diagonal_text(input_image_path, output_image_path, text, angle=45, font_path='FreeSansBold.ttf', font_size=50, opacity=0.3):
    # Open the original image
    base = Image.open(input_image_path).convert("RGBA")
    width, height = base.size

    # Create a new transparent image for the watermark
    watermark_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_layer)

    # Define the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(font_manager.get_font_names())
        print(font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
        print(f"Font not found at {font_path}. Using default font.")
        font = ImageFont.load_default()

    # Calculate position and draw the text on the transparent layer
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    position = ((width - text_width) / 2, (height - text_height) / 3)
    alpha_value = int(255 * opacity)
    draw.text(position, text, font=font, fill=(255, 0, 0, alpha_value)) # 'ms' anchors the middle of the text to the xy coordinates

    # Rotate the watermark layer
    rotated_watermark = watermark_layer.rotate(angle, expand=1)

    # Composite images: Paste the rotated watermark onto the base image
    rot_width, rot_height = rotated_watermark.size
    offset_x = (rot_width - width) // 2
    offset_y = (rot_height - height) // 2
    base.paste(rotated_watermark, (-offset_x, -offset_y), rotated_watermark)

    # Save the result
    base.save(output_image_path)

