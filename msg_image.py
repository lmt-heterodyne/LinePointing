

def mkMsgImage(pl, obsnum, txt, im, label, color='k') :

    fig = pl.figure(figsize = (10,2))
    pl.subplots_adjust(left=0, right=1, bottom=0, top=1)
    msg = '{label}: {txt}\nObsNum: {obsnum}'.format(label=label, txt=txt, obsnum=obsnum)
    print(msg)
    pl.annotate(msg, xy=(0.05,0.05), color = color, fontsize=25)
    try:
      pl.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                     bottom=False, left=False, right=False, top=False)
    except:
      pass

    for s in ['right','left','top','bottom']:
        pl.gca().spines[s].set_visible(False)
    pl.savefig(im)
    pl.close()

