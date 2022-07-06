import os
import sys
import math
import numpy as np
import traceback
from PIL import Image


def merge_focus(mags, files):
    try:
        print('merge_focus', mags, files)
        # the last file is the output file
        new_file = files[-1]
        # the two before it are the fit files
        fit_files = files[-3:-1]
        # the rest are the pointing files
        point_files = files[0:-3]
        
        print('create fit images', fit_files)
        fit_images = [Image.open(f) for f in fit_files]

        print('create point images', point_files)
        point_images = [Image.open(f) for f in point_files]

        print('figure out fit images', fit_files)
        widths, heights = zip(*(i.size for i in fit_images))
        max_width = max(widths)
        total_height = 0
        for im in fit_images:
            total_height = total_height + im.size[1]*max_width/im.size[0]

        print('figure out point images', point_files)
        num_thumbs = len(point_files)
        if(num_thumbs > 0):
            thumb_size =  max_width/num_thumbs
        else:
            thumb_size = 0
        thumb_offset = int(0.98*thumb_size)
        thumb_size = int(0.82*thumb_size)
        thumb_y_offset = point_images[0].size[1]*thumb_size/point_images[0].size[0]
        total_height += thumb_y_offset

        new_im = Image.new('RGB', (int(max_width), int(total_height)), 'white')

        x_offset = 0
        y_offset = 0
        im = fit_images[0]
        new_im.paste(im.resize((int(max_width), int(im.size[1]*max_width/im.size[0])), Image.ANTIALIAS), (x_offset,y_offset))

        x_offset = thumb_offset-thumb_size/2
        y_offset += im.size[1]*max_width/im.size[0]+thumb_y_offset
        a = np.abs(np.array(mags))
        print('a', a)
        amax = np.max(a)
        for i,im in enumerate(point_images):
            if amax != 0:
                thumb_width  = int(a[i]/amax*thumb_size)
                thumb_height = int(im.size[1]*thumb_width/im.size[0])
            else:
                thumb_height = thumb_size
            if thumb_height <= 0:
                thumb_height = 1
            new_im.paste(im.resize((int(math.ceil(thumb_width)), int(math.ceil(thumb_height))), Image.ANTIALIAS), (int(math.ceil(x_offset-thumb_width/2)), int(math.ceil(y_offset-thumb_height))))
            x_offset += thumb_offset

        x_offset = 0
        #y_offset += thumb_offset
        im = fit_images[1]
        new_im.paste(im.resize((int(math.ceil(max_width)), int(math.ceil(im.size[1]*max_width/im.size[0]))), Image.ANTIALIAS), (int(math.ceil(x_offset)), int(math.ceil(y_offset))))

        new_im.save(new_file)
        #os.system('rm -f %s'%zip(image_files))

    except Exception as e:
        print(e)
        traceback.print_exc()
        pass

if __name__ == '__main__':
    #merge_focus(sys.argv[1], sys.argv[2:-1], sys.argv[-1])
    merge_focus([2038.569733216587, 4884.6694603665765, 4592.468723424062, 2251.3286242376257], ['lmtlp_97764.png', 'lmtlp_97765.png', 'lmtlp_97766.png', 'lmtlp_97767.png', 'lf_fits_97767_1652563410021_53044.png', 'lf_model_97767_1652563410021_53044.png', 'lf_focus_97767.png'])
