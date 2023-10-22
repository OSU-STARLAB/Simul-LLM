import json


TRAIN_INPUT_PAIRS = [
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/train/txt/train-02.en', '/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/train/txt/train-02.es')
]

TEST_INPUT_PAIRS = [
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-COMMON/txt/tst-COMMON-02.en', '/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-COMMON/txt/tst-COMMON-02.es'),
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-HE/txt/tst-HE-02.en', '/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-HE/txt/tst-HE-02.es')
]

VALIDATION_INPUT_PAIRS = [
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/dev/txt/dev-02.en', '/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/dev/txt/dev-02.es')
]

OUTPUT_JSON_TRAIN = '/nfs/hpc/share/wildma/dataset_prep/datasets/2023-fall/en-es/must-c-en-es-01-train.json'
OUTPUT_JSON_TEST = '/nfs/hpc/share/wildma/dataset_prep/datasets/2023-fall/en-es/must-c-en-es-01-test.json'
OUTPUT_JSON_VALIDATION = '/nfs/hpc/share/wildma/dataset_prep/datasets/2023-fall/en-es/must-c-en-es-01-validation.json'

REJECT_STRINGS = [
    '^',
    '_',
    '***'
]

def main():

    parsed_lines = 0

    #
    # Training File
    #
    output_file_train = open(OUTPUT_JSON_TRAIN, 'w')

    is_following_an_entry = False

    for train_pair in TRAIN_INPUT_PAIRS:

        try:
            train_en_file = open(train_pair[0], 'r')
        except FileNotFoundError:
            assert False, "File not found."
        except IOError:
            assert False, "File open failed."
        try:
            train_es_file = open(train_pair[1], 'r')
        except FileNotFoundError:
            assert False, "File not found."
        except IOError:
            assert False, "File open failed."

        # Now, we go line by line
        train_en_line = train_en_file.readline()
        train_es_line = train_es_file.readline()

        # print(f'{train_en_line=} {train_es_line=}')

        while train_en_line != '' and train_es_line != '':

            if ( any(reject_string in train_en_line for reject_string in REJECT_STRINGS) or train_en_line == '\n' or
                 any(reject_string in train_es_line for reject_string in REJECT_STRINGS) or train_es_line == '\n'):

                train_en_line = train_en_file.readline()
                train_es_line = train_es_file.readline()
                continue  # This is a bad line--we're skipping it

            train_en_line = train_en_line.replace('\n', '')
            train_en_line = train_en_line.replace('"', '\\"')

            train_es_line = train_es_line.replace('\n', '')
            train_es_line = train_es_line.replace('"', '\\"')

            # if is_following_an_entry:
                # output_file.write(',')  # Values need to be comma-separated
            # output_file.write(f'{{"en": "{train_en_line}", "es": "{train_es_line}"}}')  # Commas are dealt with by the previous line
            output_file_train.write(f'{{"en": "{train_en_line}", "es": "{train_es_line}"}}\n')

            is_following_an_entry = True
            
            parsed_lines += 1
            if parsed_lines % 2500 == 0:
                print(f'Processed {parsed_lines} lines')

            train_en_line = train_en_file.readline()
            train_es_line = train_es_file.readline()
        
        train_en_file.close()
        train_es_file.close()

    # Conclude the training part
    output_file_train.close()
    print('Finished writing to the training file')

    #
    # Testing File
    #
    is_following_an_entry = False

    output_file_test = open(OUTPUT_JSON_TEST, 'w')

    for test_pair in TEST_INPUT_PAIRS:

        test_en_file = open(test_pair[0], 'r')
        test_es_file = open(test_pair[1], 'r')

        # Now, we go line by line
        test_en_line = test_en_file.readline()
        test_es_line = test_es_file.readline()

        while test_en_line != '' and test_es_line != '':

            if ( any(reject_string in test_en_line for reject_string in REJECT_STRINGS) or test_en_line == '\n' or
                 any(reject_string in test_es_line for reject_string in REJECT_STRINGS) or test_es_line == '\n'):

                test_en_line = test_en_file.readline()
                test_es_line = test_es_file.readline()
                continue  # This is a bad line--we're skipping it

            test_en_line = test_en_line.replace('\n', '')
            test_en_line = test_en_line.replace('"', '\\"')

            test_es_line = test_es_line.replace('\n', '')
            test_es_line = test_es_line.replace('"', '\\"')

            # if is_following_an_entry:
                # output_file.write(',')  # Values need to be comma-separated
            # output_file.write(f'{{"en": "{test_en_line}", "es": "{test_es_line}"}}')  # Commas are dealt with by the previous line
            output_file_test.write(f'{{"en": "{test_en_line}", "es": "{test_es_line}"}}\n')

            is_following_an_entry = True
            
            parsed_lines += 1
            if parsed_lines % 2500 == 0:
                print(f'Processed {parsed_lines} lines')

            test_en_line = test_en_file.readline()
            test_es_line = test_es_file.readline()
        
        test_en_file.close()
        test_es_file.close()
    
    output_file_test.close()
    print('Finished writing to the testing file')

    #
    # Validation File
    #
    output_file_validation = open(OUTPUT_JSON_VALIDATION, 'w')

    for validation_pair in VALIDATION_INPUT_PAIRS:

        validation_en_file = open(validation_pair[0], 'r')
        validation_es_file = open(validation_pair[1], 'r')

        # Now, we go line by line
        validation_en_line = validation_en_file.readline()
        validation_es_line = validation_es_file.readline()

        while validation_en_line != '' and validation_es_line != '':

            if ( any(reject_string in validation_en_line for reject_string in REJECT_STRINGS) or validation_en_line == '\n' or
                 any(reject_string in validation_es_line for reject_string in REJECT_STRINGS) or validation_es_line == '\n'):

                validation_en_line = validation_en_file.readline()
                validation_es_line = validation_es_file.readline()
                continue  # This is a bad line--we're skipping it

            validation_en_line = validation_en_line.replace('\n', '')
            validation_en_line = validation_en_line.replace('"', '\\"')

            validation_es_line = validation_es_line.replace('\n', '')
            validation_es_line = validation_es_line.replace('"', '\\"')

            output_file_validation.write(f'{{"en": "{validation_en_line}", "es": "{validation_es_line}"}}\n')

            is_following_an_entry = True
            
            parsed_lines += 1
            if parsed_lines % 2500 == 0:
                print(f'Processed {parsed_lines} lines')

            validation_en_line = validation_en_file.readline()
            validation_es_line = validation_es_file.readline()
        
        validation_en_file.close()
        validation_es_file.close()
    
    output_file_validation.close()
    print('Finished writing to the validationing file')



if __name__ == "__main__":
    main()
