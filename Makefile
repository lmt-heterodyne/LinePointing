clean:
	rm -f *~ *.pyc *.png *.json *.html *.log
	rm -fr __pycache__
	rm -f vlbi1mm_tsys.html


test1:
	@echo Uses obsnum 83578, 83579, 83580, 83581, 83582
	python allfocus.py

test2:
	@echo Uses obsnum 83578, 83579, 83580, 83581, 83582
	python linefocus.py 83578,83579,83580,83581,83582

test3:
	@echo Uses obsnum 78003 78004 76406 76085 78065 78091
	python linepoint.py 92984
