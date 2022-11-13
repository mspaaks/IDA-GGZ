# IDA-GGZ

These scripts are part of the research project Intellectual Disability Alert, in which a first draft of a machine learning algorithm to identify patients with intellectual disabilities is created. In the scripts, CSV files are created which are needed in the following scripts. Therefore, the path to these files should be provided. 

The files should be run in the following order: 
-Files loading the data: p02psyf_load, p03anam_load, p04medic_load, p05lich_load, p06labbe_load, p09gaf_load, dsm_dataload. 
- To be able to fill the NaN's, 'replace_missing' should be run. 
- Combine all data frames from the dataloads using: 'combine_dataload'
- Balance datasets on age and gender using: 'Matchen voor Marije.R'
- Use 'impute_data' to be able to use the k-NN imputer
- Create the machine learning model running: 'machine_learning'
