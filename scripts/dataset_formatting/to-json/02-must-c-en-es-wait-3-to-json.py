from typing import List
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

OUTPUT_JSON_TRAIN = '/nfs/hpc/share/wildma/dataset_prep/datasets/2023-fall/en-es/wait3/must-c-en-es-01-train.json'
OUTPUT_JSON_TEST = '/nfs/hpc/share/wildma/dataset_prep/datasets/2023-fall/en-es/wait3/must-c-en-es-01-test.json'
OUTPUT_JSON_VALIDATION = '/nfs/hpc/share/wildma/dataset_prep/datasets/2023-fall/en-es/wait3/must-c-en-es-01-validation.json'

REJECT_STRINGS = [
    '^',
    '_',
    '***'
]


def words_to_sntc(wordlist: List[str]) -> str:
    """
    :param wordlist: List[str] of words, may include punctuation
    """
    return ' '.join(wordlist).strip()


def format_wait_k_multi_sentences(source: str, target: str, k: int, 
                                  eos_token: str = '<|endoftext|>',
                                  empty_repr: str = '') -> List[List[str]]:
    """
    :param source: 1+ sentences in a source lange
    :param target: 1+ sentences that are a translation of the source
    :param k: how many source words to wait before predicting the next translated word 
    :param eos_token: EOS token added as the output when inputted with a complete translation
    :param empty_repr: Representation of what an "empty" value should look like

    What this function does: given a source sentence and target translation, formats multiple
        wait-k distributions of prompt content. 

    The output is a list of tuples in the form [current_source, current_target, target_token]    
    """
    
    wait_k_prompts = []
        
    src_words = source.split(' ')
    tgt_words = target.split(' ')
    src_wordcount = len(src_words)
    tgt_wordcount = len(tgt_words)
    assert(src_wordcount > 0)
    
    for i in range(tgt_wordcount):  # For every target word, pair it with the source words k words ahead

        current_source = words_to_sntc(src_words[0:min(src_wordcount + 1, i + (k - 1) + 1)])

        current_translation = words_to_sntc(tgt_words[0:i]) or empty_repr

        next_word = tgt_words[i]

        new_entry = [current_source, current_translation, next_word]
        wait_k_prompts.append(new_entry)

        # # We're at the last target word, but source still has words left after k
        # if i + 1 == tgt_wordcount and i + k < src_wordcount:  

        #     # If want to include an example that has the fullness of the source--
        #     full_source = source
        #     current_translation = words_to_sntc(tgt_words[:-1])
        #     last_word = tgt_words[-1]
            # new_entry = (full_source, current_translation, last_word)
            # wait_k_prompts.append(new_entry)

    # Finally, include an example with an EOS Token
    new_entry = [source, target, eos_token]
    wait_k_prompts.append(new_entry)

    return wait_k_prompts


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
            train_es_line = train_es_line.replace('\n', '')

            for wait3_output in format_wait_k_multi_sentences(train_en_line, train_es_line, k=3):

                wait3_output[0] = wait3_output[0].replace('"', '\\"')
                wait3_output[1] = wait3_output[1].replace('"', '\\"')
                wait3_output[2] = wait3_output[2].replace('"', '\\"')

                output_file_train.write(f'{{"current_source": "{wait3_output[0]}", "current_target": "{wait3_output[1]}", "target_token": "{wait3_output[2]}"}}\n')
            
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
            test_es_line = test_es_line.replace('\n', '')

            for wait3_output in format_wait_k_multi_sentences(test_en_line, test_es_line, k=3):

                wait3_output[0] = wait3_output[0].replace('"', '\\"')
                wait3_output[1] = wait3_output[1].replace('"', '\\"')
                wait3_output[2] = wait3_output[2].replace('"', '\\"')

                output_file_test.write(f'{{"current_source": "{wait3_output[0]}", "current_target": "{wait3_output[1]}", "target_token": "{wait3_output[2]}"}}\n')
            
            parsed_lines += 1
            if parsed_lines % 2500 == 0:
                print(f'Processed {parsed_lines} lines')

            test_en_line = test_en_file.readline()
            test_es_line = test_es_file.readline()
        
        test_en_file.close()
        test_es_file.close()
    
    # Conclude testing part and close file
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
            validation_es_line = validation_es_line.replace('\n', '')

            for wait3_output in format_wait_k_multi_sentences(validation_en_line, validation_es_line, k=3):

                wait3_output[0] = wait3_output[0].replace('"', '\\"')
                wait3_output[1] = wait3_output[1].replace('"', '\\"')
                wait3_output[2] = wait3_output[2].replace('"', '\\"')

                output_file_validation.write(f'{{"current_source": "{wait3_output[0]}", "current_target": "{wait3_output[1]}", "target_token": "{wait3_output[2]}"}}\n')
            
            parsed_lines += 1
            if parsed_lines % 2500 == 0:
                print(f'Processed {parsed_lines} lines')

            validation_en_line = validation_en_file.readline()
            validation_es_line = validation_es_file.readline()
        
        validation_en_file.close()
        validation_es_file.close()
    
    # Conclude validation part and close file
    output_file_validation.close()
    print('Finished writing to the validation file')


if __name__ == "__main__":
    main()
