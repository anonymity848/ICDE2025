Readme (Interactive Search with Reinforcement Learning)
=========================
This package contains all source codes for 
a. Algorithm EA 
	1. It is an exact algorithm designed by us. 
	2. The code is in file lowRL.py.
b. Algorithm AA 
	1. It is an approximate algorithm designed by us. 
	2. The code is in file highRL.py.
c. Algorithm AA-Random 
	1. It is an approximate algorithm designed by us with randomly selected action space. 
	2. The code is in file highRL.py.
d. Algorithm UH-Random
	1. It is the SOTA algorithm used for comparison. 
	2. The code is in file uh.py.
e. Algorithm UH-Simplex 
	1. It is an existing algorithm used for comparison. 
	2. The code is in folder uh.py.
f. Algorithm SinglePass
	1. It is an existing algorithm used for comparison. 
	2. The code is in folder single_pass.py.

Usage Step
==========
a. Package
	Please install the packages needed by the code, e.g., pytorch, swiglpk, matplotlib, etc. 
	
b. Execution
	The command has several parameters.
	'''
	python main.py (1)Algorithm_name (2)Dataset_name (3)Epsilon (4)Training_size (5)Action_space_size (6)Utility_vector_<u[1], u[2], ..., u[d]>
	'''
	E.g., highRL 4dSkyline 0.1 10000 5 0.1 0.1 0.1 0.1

c. Results
	The results will be shown in the console. 

