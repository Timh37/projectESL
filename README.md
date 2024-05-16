# projectESL
**projectESL** is a python repository for analyzing extreme sea levels from various sources and using the resulting extreme value distributions to project frequency amplification factors due to projected sea-level change.
The repository can do (stationary) extreme value analysis of high-frequency tide gauge data from the GESLA2 and 3 databases and from hydrodynamic model output (GTSM). It also supports ingestion of predefined extreme value distribution parameters or return curves from a number of previous studies, varying in spatial coverage. The user has the option to combine the information on extreme sea levels with mean sea-level projections to compute amplification factors or their timing. 

## Instructions
First, configure **projectESL** by changing the configuration file ```config.yml```. Make sure that the paths to the files required for the configured settings point to the right directories and that the required data has been downloaded and prepared correctly beforehand (see the tables below for further instructions). Then, run **projectESL** by executing ```run.py```. This will first process the extreme value data, and depending on the configured settings, then combine extreme value distributions with the provided sea-level projections to compute projections of extreme sea levels due to future sea-level change. For the last step, a dask distributed cluster is started to process different locations in parallel and hande out-of-memory data.


## Options & required data
To run **projectESL**, an input xarray file holding the latitude and longitude coordinates of the desired input locations must always be specified (under <em>input_locations</em> in ```config.yml```).
If configuring the program to project amplification factors and/or their timing in addition to producing extreme sea-level distributions, this input file also needs to contain sea-level projections. As **projectESL** was designed to be compatible with the FACTS framework (<https://github.com/radical-collaboration/facts>),
the input dataset is expected to be formatted according to the format of FACTS regional sea-level projections output. A helper function ```generate_sites_input.py``` is provided to help prepare the input file.

**projectESL** allows the user to derive extreme sea-level distributions from various data sources. The following table details these options and the data required for them:

| **Input otion** | **Required data**                                                           |
|-----------------|-----------------------------------------------------------------------------|
| gesla2          | Available from <https://gesla787883612.wordpress.com/gesla2/>               |
| gesla3          | Available from <https://gesla787883612.wordpress.com/downloads/>            |
| vousdoukas2018  | input/Vousdoukas2018_Historical_ERA-INTERIM_wl.nc (based on surge only)     |
| virezci2020     | input/Kirezci2020_gpd.csv                                                   |
| hermans2023     | input/Hermans2023_gesla3_gpd_daily_max_potATS_Solari.nc                     |
| coast-rp        | Full dataset from Dullaart et al. (2021), available upon request of authors |

The user has the option to use constant reference frequencies for the computation of amplification factors, to use estimated flood protection standards based on historical cost-benefit modeling, or to use custom flood protection standards:


| **Input otion** | **Required data**                                                           |
|-----------------|-----------------------------------------------------------------------------|
| diva            | Available upon request from Daniel Lincke                                   |
| flopros         | Available from <https://zenodo.org/records/3475120>                         |
| custom          | custom data, for example input/flopros_estimates_at_coastrp_locations.nc    |

After obtaining the relevant data, the user needs to set the paths to these data in ```config.yml```.

## Dependencies
- numpy
- pandas
- xarray
- dask
- zarr
- netcdf4
- scipy
- tqdm
- yaml
- geopandas (if using 'diva' or 'flopros' as reference return frequencies)
- matplotlib (only for visualization)
- cartopy (only for visualization)
- cmocean (only for visualization)
 
 

