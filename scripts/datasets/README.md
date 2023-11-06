The upload_cleaned_mustc.py script is used to clean MuST-C verbal text translations and upload the cleaned translations to anyone's 
    own dataset repository in HuggingFace, uploading either full translations and/or translations partitioned into wait-k segments.


Notes on running the dataset creation script:

    1. A MuST-C dataset must be downloaded locally somewhere, and the path to this dataset must be included in the --must-c-dir argument

    2. When running the script, it must be indicated that either a. that the script should create full translation dataset with 
        the -f flag or b. the script should create one or more partitioned wait-k datasets with the --k-constants flag
       
    3. Temporary JSON files are created locally to be uploaded to the HuggingFace, and then deleted once the script is over.
        Each dataset requires about 1.2 gigabytes of JSON files to be temporary created. If this it too much for your local
        directory, indicate a different directory for them to be created with the --temporary-output-dir argument

    4. Run 

            python upload_cleaned_mustc.py --help

        To see all available arguments
        
    Current available translation types:

        1. en <-> es
        2. en <-> de
        3. en <-> fr


Example Calls:

    python upload_cleaned_mustc.py -f --k-constants 2 4 --translation-type en-de --must-c-dir 
    /nfs/hpc/share/mydir/downloaded_datasets/must-c-en-de 
        
    python upload_cleaned_mustc.py -r -f --k-constants 3 5 7 9 --translation-type en-fr --must-c-dir 
    /nfs/hpc/share/mydir/must-c-downloads/en-fr 
