# LinePointing

Pointing code for Wares based data

## Python packages we need:


    PIL
	 
should go into **requirements.txt**

	 

# Examples

## allfocus (test1)

for obsnums: 83578,83579,83580,83581,83582

      python allfocus.py 

## linefocus

accepts a comma separated list of obsnums

      python linefocus.py obsnums
	  
for example

      python linefocus.py 83578,83579,83580,83581,83582
      
	  

## linepoint

For beam maps. E.g. for obsnum 76406

      python linepoint.py 76406

Single character arguments allow a shortcut for certain benchmark obsnums:

      c   78003     Chi_Cyg cal 
      g   78004     Chi_Cyg grid
      m   76406     IRC otf 
      p   76085     MARS otf
      b   78065     BS
      9pt 78091     PS

The 2021 season started with 

    92984 - beam map at peak focus
    92986 - beam map at +1.75mm focus
    92988 - beam map at -1.75mm focus

and a set with a deliberate astigmatism set into the dish:

    92992 - beam map at peak focus
    92994 - beam map at +1.75mm focus
    92996 - beam map at -1.75mm focus
    
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

The command **lmtinfo.py $DATA_LMT 79448** would give a bit more info.

# History

* original code, developed under CVS
* Imported from CVS to git (November 2021)
