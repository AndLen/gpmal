# gp-mal

GP-MaL entry point is tests.featureLearn.FLNeighboursFG (as proposed in EuroGP '19)
GP-MaL using ALL neighburs entry point is tests.featureLearn.FLNeighbours  

Uses JUnit via tests.Tests to run on a range of different datasets automatically, e.g. a_irisTest()  

Alternatively, there is an exectuable jar that can be used at target/gp-mal-eurogp-19-bin.jar in the form:
java -jar gp-mal-eurogp-19-bin.jar "tests.featureLearn.FLNeighboursFG#a_irisTest" for the Iris dataset. The "a_irisTest" corresponds to the dataset being tested, which is stored inside the jar (see tests.Tests). 

I can probably find a nicer way so datasets can be provided directly from the command line if the above is too obnoxious. let me know.

datasets are in the datasets/ folder and use a variation on CSV format. You may want to adapt this...  

e.g. iris.data:
classLast,4,3,comma  
5.1,3.5,1.4,0.2,Iris-setosa  
4.9,3.0,1.4,0.2,Iris-setosa  
4.7,3.2,1.3,0.2,Iris-setosa  
4.6,3.1,1.5,0.2,Iris-setosa  
5.0,3.6,1.4,0.2,Iris-setosa  
....  

classLast: the class label is last (vs classFirst);  
4: 4 features;  
3: 3 classes;  
comma: comma-separated file (vs space for space-separated or tab for tab-separated).  
