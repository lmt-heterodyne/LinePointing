import os
import sys
import traceback
from PIL import Image


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


    
