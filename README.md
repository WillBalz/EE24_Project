To "reproduce our results" you can just run jaywalk_model.py and play around with it. 
As mentioned in the presentation, the best use case is small clusters of people.
For example, run it on a group of 10 people for 15 seconds with initial wait time 6 seconds and exponential decay parameter 2.
That was a common one we tested.
Other possibilites to exaggerate the effect are to run it with larger amounts of people, and see how they quickly flock.

The other file, EE24_Sim.py, is the graphical simulation using ASCII art. We didn't end up using it for analysis, only visualization.
It was much more complicated to illustrate the simulation with the more complex and realistic models. 

Note: We'll totally acknowledge that we did plenty of vibecoding + revision. Though it's about jaywalking, much of this stuff
was pretty complex to render. We figured the fruit of the project lay in the analysis and thought, and wanted to make sure we weren't
limiting the possibilities with time/coding constraints.
