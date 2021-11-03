clean:
	rm -f *~ *.pyc *.png
	rm -fr __pycache__
	rm -f vlbi1mm_tsys.html


test1:
	@echo Uses obsnum 83578, 83579, 83580, 83581, 83582
	python allfocus.py

test2:
	@echo Uses obsnum 83578, 83579, 83580, 83581, 83582
	python linefocus.py 83578,83579,83580,83581,83582
