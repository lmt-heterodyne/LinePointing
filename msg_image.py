import textwrap

def mkMsgImage(obsnum, txt, im, label, color='k', bg_img=None) :
    import matplotlib.pyplot as pl

    if bg_img is not None:
        import matplotlib.image as img
        bg = img.imread(bg_img)
        fig = pl.figure(figsize = (bg.shape[1]/100, bg.shape[0]/100))
        ax = pl.axes((0,0,1,1)) #axes over whole figure
        ax.imshow(bg)
        pl.imshow(bg)
        msg = '{label}:\n{txt}\nObsNum: {obsnum}'.format(label=label, txt=txt, obsnum=obsnum)
        print(msg)
        ax.text(bg.shape[1]/4, bg.shape[0], msg, color = color, fontsize=50, rotation=90)#, wrap=True)
    else:
        fig = pl.figure(figsize = (10,10))
        pl.subplots_adjust(left=0, right=1, bottom=0, top=1)
        txt = '\n'.join(textwrap.fill(text=x, width=45) for x in txt.split('\n'))
        #txt = textwrap.fill(text=txt, width=45)
        msg = '{label}:\n{txt}\nObsNum: {obsnum}'.format(label=label, txt=txt, obsnum=obsnum)
        print(msg)
        pl.annotate(msg, xy=(0.05,0.05), color = color, fontsize=25)#, wrap=True)
    try:
      pl.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                     bottom=False, left=False, right=False, top=False)
    except:
      pass

    for s in ['right','left','top','bottom']:
        pl.gca().spines[s].set_visible(False)
    #pl.tight_layout()
    pl.savefig(im)
    pl.close()

