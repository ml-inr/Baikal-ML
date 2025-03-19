1. Set paths to mu and nu root_files in given proportion. 
2. Set filter params for processors
3. Init Processors
    1. processor for mu reads mu_paths by chunks of about 20. Calcs statistics. 
    2. processor for nuatm reads nu_paths by chunks of about 20. Calcs statistics.
    3. processor for nu2 reads nu_paths by chunks of about 20. Calcs statistics. 
4. we estimate mu/nuatm/nu2 ratio in data. If bad --> ReDo proportion in 1.
5. Norming params estimated. X,Y,Z -- some standard cluster, t,Q -- from data. 

6. Paths and their splitting to train/test/val are chosen.
7. Generator, that returns shuffled events in batches (maintainig particles proportions) from shuffled paths.
    
