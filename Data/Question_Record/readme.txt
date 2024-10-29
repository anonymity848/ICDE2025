These folders record the interactive questions asked by different algorithms on different datasets. 

The name of each folder consists of three parts. 
1. The first part shows the name of the algorithm. 
2. The second part shows the name of the dataset. 
3. The third part shows the value of the parameter epsilon. 
For example, you might see "AA_20d_0.1". It means this folder records interactive questions asked by algorithm AA on the synthetic dataset with 20 dimensions, where epsilon is 0.1. 

In each folder, you could see several files. Each file consists of three parts. 
1. The utility vector used for this interaction process. 
2. The interactive questions.
3. The total number of questions asked and the final returned tuple. 

For example, you might see a similar content as follows. 
==============================================================================
The generated utility vector is u = [0.04935112 0.35124538 0.31088880 0.28851470] 


Question 1: 
+---------------+---------------+---------------+---------------+---------------+
|     Tuple     |  Attribute 1  |  Attribute 2  |  Attribute 3  |  Attribute 4  |
+---------------+---------------+---------------+---------------+---------------+
|    Tuple 1    |    0.185663   |    0.745335   |    0.861635   |    0.530325   |
+---------------+---------------+---------------+---------------+---------------+
|    Tuple 2    |    0.42503    |    0.762416   |    0.970842   |    0.164754   |
+---------------+---------------+---------------+---------------+---------------+
The user selects Point 1 as his/her preferred one.

......(There are many interactive questions)......

The finally returned tuple: 
+---------------+---------------+---------------+---------------+---------------+
|     Tuple     |  Attribute 1  |  Attribute 2  |  Attribute 3  |  Attribute 4  |
+---------------+---------------+---------------+---------------+---------------+
|     Tuple     |    0.072617   |    0.871462   |    0.946912   |    0.627919   |
+---------------+---------------+---------------+---------------+---------------+
The total number of questions asked: 9 

==============================================================================
This content shows:
1. The utility vector used for this interaction process is u = [0.04935112 0.35124538 0.31088880 0.28851470]. 
2. The interactive question consists of two tuples. 
	The first tuple is [0.185663, 0.745335, 0.861635, 0.530325].
	The second tuple is [0.42503, 0.762416, 0.970842, 0.164754]. 
	For this question, the user prefers tuple 1 to tuple 2. 
3. The final return tuple is [0.072617, 0.871462, 0.946912, 0.627919] and the total number of questions asked is 9. 





