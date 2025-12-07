In folder_0 you can find the output of Climatol that are flag.csv and series.csv. 
Climatol is an R package holding functions for quality control, homogenization and missing data in-filling of climatological series. 
series.csv contains the values of the AEMET observation.
flags.csv contains values 0,1,2 and they identify missing data, good data, noisy data respectivly.
Climatol_to_grid_CERRA,py process the AEMETER observation and convert them in CERRA grid for having both datasets matched.
checking_aemet_projecton2CERRA.py check 
Join_data.ipynb is a notebook that concatanates the annual datas and connect each AEMET station to a pixel 

In folder_1 you can find some scripts for the dowloading of the yearly CERRA dataset and scripts for extract the daily averages from them and 
merge them in a single file.

In folder_2 you can find some scripts for slice the AEMET dataset converted to LCC and for create a mask with 1 where we have observation
and 0 where we don't have observations.

In folder 4 you can find the scripts for splitting AEMET and CERRA datasets and the mask in peninsula and Canaria

In folder 5 you can find the scripts for the splitting in train, valitation and test sets.
