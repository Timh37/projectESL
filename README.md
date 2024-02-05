# projectESL
**projectESL** is a python repository for analyzing extreme sea levels and projecting the magnitude and timing of their frequency amplification future sea-level rise. 
The repository is capable of doing (stationary) extreme value analysis of high-frequency tide gauge data from the GESLA2 and 3 databases and also supports reading in predefined extreme value distribution parameters or return curves from a number of previous studies, varying in spatial coverage.
The user can also choose to combine the information on extreme sea levels with mean sea-level projections to compute amplification factors or their timing. 

## Instructions
Before running **projectESL**, the user must indicate which input data to use for the ESL analysis, where to find the required input data and what results to store where in the configuration file "config.yml". 
A netcdf or zarr file with sites at which to compute ESLs and potential other results is always required, and functions to help generate this input will be made available from '''utils.py'''. Providing sea-level projections at these sites is optional but required when outputting amplification factors.
After configuring the run, the program can simply be ran by calling the run-script "run.py", e.g., from the command line: '''python run.py'''. 

The table below lists the input options currently implemented: 

 
 

