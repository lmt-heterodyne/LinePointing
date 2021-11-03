# LinePointing

Pointing code for Wares based data

## Python packages we need:


    PIL
	 
should go into **requirements.txt**

	 

# Examples

## allfocus

for obsnums: 83578, 83579, 83580, 83581, 83582

## linefocus

accepts a list of obsnums

## linepoint

for obsnum 76406

      python linepoint.py 76406
	  
no roach files found

## lmtlp_reduce_cli

talks to lmtlp_reduce_srv/

      python lmtlp_reduce_cli obsnum opt line_list baseline_list tsys

## lmtlp_reduce

      python lmtlp_reduce obsnum[;{key1}:{val1};{key2}:{val2};...]
	  
## lmtlp_reduce_srv

server code, for lmtlp_reduce_cli. Can only work at LMT ?

## merge_focus

      python merge_focus.py  <mags> <files>   <????>
	  
something odd about 2 vs. 3 arguments

## test

      python test.py obsnum
	  
Lists the obsnum, receiver, obspgm, e.g.

    python test.py 79448
      79448
      obsnum 79448
      receiver Sequoia
      obspgm Map



# History

* original code, developed under CVS
* Imported from CVS to git (November 2021)
