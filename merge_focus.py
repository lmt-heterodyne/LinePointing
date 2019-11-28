import os
import sys
import numpy as np
from PIL import Image

def merge_focus(mags, files):
    try:
	print 'merge_focus', mags, files
        # the last file is the output file
        new_file = files[-1]
        # the two before it are the fit files
        fit_files = files[-3:-1]
        # the rest are the pointing files
        point_files = files[0:-3]
        
	print 'create fit images', fit_files
	fit_images = map(Image.open, fit_files)

	print 'create point images', point_files
	point_images = map(Image.open, point_files)

	print 'figure out fit images', fit_files
	widths, heights = zip(*(i.size for i in fit_images))
	max_width = max(widths)
	total_height = 0
	for im in fit_images:
	    total_height = total_height + im.size[1]*max_width/im.size[0]

        print 'figure out point images', point_files
        num_thumbs = len(point_images)
        if(num_thumbs > 0):
            thumb_size =  max_width/num_thumbs
        else:
            thumb_size = 0
        total_height += thumb_size
        thumb_offset = int(0.98*thumb_size)
        thumb_size = int(0.82*thumb_size)

	new_im = Image.new('RGB', (max_width, total_height), 'white')

	x_offset = 0
	y_offset = 0
	im = fit_images[0]
	new_im.paste(im.resize((max_width, im.size[1]*max_width/im.size[0]), Image.ANTIALIAS), (x_offset,y_offset))

        x_offset = thumb_offset-thumb_size
	y_offset += im.size[1]*max_width/im.size[0]
        a = np.array(mags)
        amax = np.max(a)
        for i,im in enumerate(point_images):
            if amax != 0:
                thumb_height = int(a[i]/amax*thumb_size)
            else:
                thumb_height = thumb_size
	    new_im.paste(im.resize((int(thumb_height), int(thumb_height)), Image.ANTIALIAS), (x_offset+thumb_size/2-thumb_height/2, y_offset+thumb_offset-thumb_height))
            x_offset += thumb_offset

        x_offset = 0
	y_offset += thumb_offset
	im = fit_images[1]
	new_im.paste(im.resize((max_width, im.size[1]*max_width/im.size[0]), Image.ANTIALIAS), (x_offset,y_offset))

	new_im.save(new_file)
	#os.system('rm -f %s'%zip(image_files))

    except Exception as e:
	print e
	pass

if __name__ == '__main__':
    merge_focus(sys.argv[1], sys.argv[2:-1], sys.argv[-1])
