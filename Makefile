clean:
	rm -f *~ *.pyc *.png
	rm -fr __pycache__

rsync:
	rsync -avz --exclude '__pycache__' --exclude '*.png' --exclude '*~' --exclude '*.pyc' ~/LinePointing3/ lmtmc@mnc-mx:LinePointing3
	rsync -avz --exclude '__pycache__' --exclude '*.png' --exclude '*~' --exclude '*.pyc' ~/LinePointing3/ lmtmc@wares:LinePointing3
	rsync -avz --exclude '__pycache__' --exclude '*.png' --exclude '*~' --exclude '*.pyc' ~/LinePointing3/ lmtmc@pico:LinePointing3




