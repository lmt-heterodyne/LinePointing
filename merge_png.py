import os
import sys
import traceback
from PIL import Image


def merge_png(image_files, newfile):
    try:
        print ('merge_png', image_files, newfile)
        images = list(map(Image.open, image_files))
        widths, heights = zip(*(i.size for i in images))

        max_width = max(widths)
        total_height = 0
        for im in images:
            total_height = total_height + im.size[1]*max_width/im.size[0]

        new_im = Image.new('RGB', (int(max_width), int(total_height)))

        x_offset = 0
        y_offset = 0
        for im in images:
            try:
                new_im.paste(im.resize(
                    (int(max_width), int(im.size[1]*max_width/im.size[0])), Image.ANTIALIAS), (int(x_offset), int(y_offset)))
            except:
                new_im.paste(im.resize(
                    (int(max_width), int(im.size[1]*max_width/im.size[0])), Image.LANCZOS), (int(x_offset), int(y_offset)))
                
            y_offset += im.size[1]*max_width/im.size[0]

        new_im.save(newfile)
        #os.system('rm -f %s'%zip(image_files))

    except Exception as e:
        print (e)
        traceback.print_exc()
        pass


    
