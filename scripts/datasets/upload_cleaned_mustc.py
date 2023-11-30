from prep_functions import safe_open_w, format_wait_k_multi_sentences
import mustc_cleaning_functions
from typing import Dict, List
import os
import shutil
import argparse
from datetime import datetime
import huggingface_hub
import datasets


ALLOWED_PAIRS = [
    ('en', 'de'),
    ('en', 'es'),
    ('en', 'fr')
]

TERMINATING_SYMBOL_CODES = {

    'falcon': '<|endoftext|>',
    'llama': '</s>'
}

CLEANING_FUNCTIONS = {

    'en': mustc_cleaning_functions.clean_en,
    'es': mustc_cleaning_functions.clean_es,
    'de': mustc_cleaning_functions.clean_de,
    'fr': mustc_cleaning_functions.clean_fr
}


def language_pair_is_allowed(src_lang_code: str, tgt_lang_code: str) -> bool:

    if src_lang_code == tgt_lang_code:
        return False 

    for pair in ALLOWED_PAIRS:

        if src_lang_code in pair and tgt_lang_code in pair:
            return True

    return False


def populate_dictionary(root, files, dict: Dict, lang_code: str) -> None:
    """
    Looks through the given files to see if it can populate the directionary with their paths

    lang_code is ISO codes such as 'de', 'es', 'en', ...
    """

    for content_type in ['train', 'tst-COMMON', 'tst-HE', 'dev']:

        if f'{content_type}.{lang_code}' in files:
            dict[content_type] = os.path.join(root, f'{content_type}.{lang_code}')

            if f'{content_type}-cleaned.{lang_code}' in files:
                dict['precleaned-must-c-data'][content_type] = os.path.join(root, f'{content_type}-cleaned.{lang_code}')


def cleaned_filepath(path: str) -> str:

    # Split the file path into directory, base, and extension
    dir_name, file_name = os.path.split(path)
    base_name, extension = os.path.splitext(file_name)
    
    # Append '-cleaned' to the base name
    new_base_name = f'{base_name}-cleaned'
    
    # Reconstruct the path with the new base name
    new_file_path = os.path.join(dir_name, new_base_name + extension)
    
    return new_file_path


def clean_mustc_data(lang_code: str, filepaths_dict: Dict) -> int:
    """
    Makes four cleaned language files and returns how many files were created
    """

    files_outputted = 0

    for content_type in ['train', 'tst-COMMON', 'tst-HE', 'dev']:

        lang_data = ''
        language_filepath = filepaths_dict[content_type]

        with open(language_filepath, 'r') as f:
            lang_data = f.read()

        lang_data = CLEANING_FUNCTIONS.get(lang_code)(lang_data)

        cleaned_output_path = cleaned_filepath(language_filepath)

        with open(cleaned_output_path, 'w') as f:
            f.write(lang_data)

            filepaths_dict['precleaned-must-c-data'][content_type] = cleaned_output_path
            files_outputted += 1

    return files_outputted


def write_wait_k_to_file(source: str, target: str, output_file, k_constant: int, eos_token: str) -> None:

    for wait_k_output in format_wait_k_multi_sentences(source, target, k=k_constant, eos_token=eos_token):

        output_file.write(f'{{"current_source": "{wait_k_output[0]}", "current_target": "{wait_k_output[1]}", "target_token": "{wait_k_output[2]}"}}\n')

    
def create_translation_jsons(k_constants: List | None, temporary_output_dir: str, source_filepaths: Dict, target_filepaths: Dict,
                 src_lang_code: str, tgt_lang_code: str, eos_token: str, generate_full_translation: bool) -> None:

    REJECT_STRINGS = [
        '^',
        '_',
        '***',
        '*',
        '<',
        '>',
        '+'
    ]

    if k_constants is None:
        k_constants = []

    # Create our temporary output dir if it doesn't exist
    output_dir = os.path.dirname(temporary_output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    todo_list = {
        'train': ['train'],
        'test': ['tst-COMMON', 'tst-HE'],
        'validation': ['dev']
    }

    parsed_lines = 0

    print('Beginning to process datasets...')

    # For every train, test, and validation category:
    for json_type, content_type_list in todo_list.items():

        # Open a filepath for every k-value we want to create a json for
        output_files = []
        for k in k_constants:

            filepath = f'wait{str(k).zfill(2)}/must-c-{src_lang_code}-{tgt_lang_code}-{json_type}.json'
            full_output_json_path = os.path.join(temporary_output_dir, filepath)

            output_files.append(safe_open_w(full_output_json_path))
        
        if generate_full_translation:
            filepath = f'full/must-c-{src_lang_code}-{tgt_lang_code}-{json_type}.json'
            full_output_json_path = os.path.join(temporary_output_dir, filepath)

            full_mt_output_file = safe_open_w(full_output_json_path)

        # Now, for every file we have available to use for a category: 
        # (example, there is tst-COMMON and tst-HE available for test)
        for content_type in content_type_list:

            src_file = open(source_filepaths['precleaned-must-c-data'][content_type], 'r')
            tgt_file = open(target_filepaths['precleaned-must-c-data'][content_type], 'r')

            # Go line by line and one at a time, add that line's wait-k data to the json
            src_line = src_file.readline()
            tgt_line = tgt_file.readline()

            # Read every line we have available in the file
            while src_line != '' and tgt_line != '':

                if ( any(reject_string in src_line for reject_string in REJECT_STRINGS) or src_line == '\n' or
                    any(reject_string in tgt_line for reject_string in REJECT_STRINGS) or tgt_line == '\n'):

                    src_line = src_file.readline()
                    tgt_line = tgt_file.readline()
                    continue  # This is a bad line--we're skipping it

                # Pre-clean the lines
                src_line = src_line.replace('\n', '').replace('"', '\\"')
                tgt_line = tgt_line.replace('\n', '').replace('"', '\\"')

                # Write to the full translation file
                if generate_full_translation:
                    full_mt_output_file.write(f'{{"{src_lang_code}": "{src_line}", "{tgt_lang_code}": "{tgt_line}"}}\n')

                # Write to the wait_k files
                for i in range(len(k_constants)):
                    write_wait_k_to_file(src_line, tgt_line, output_files[i], k_constants[i], eos_token)
                
                parsed_lines += 1
                if parsed_lines % 50000 == 0:
                    print(f'Processed {parsed_lines} lines')

                src_line = src_file.readline()
                tgt_line = tgt_file.readline()
            
            src_file.close()
            tgt_file.close()
        
        for opened_file in output_files:
            opened_file.close()
        output_files = []

        if generate_full_translation:
            full_mt_output_file.close()


def initialize_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()

    parser.add_argument('-r', action='store_true', default=False,
                        help='Flag to regenerate new cleaned must-c files even if old ones already exist')

    parser.add_argument('-f', action='store_true', default=False,
                        help='Flag to upload a full dataset without being partitioned to wait-k format, just in whole translation pairs')

    parser.add_argument('--k-constants', type=int, nargs='+', required=False,
                        help='k-constants you wish to make wait-k datasets for, example format: "3 4 5 7" or "2"')

    parser.add_argument('--terminating-symbol-type', type=str, default='falcon',
                        help='Used solely for wait-k translations. Represents which terminating symbol to be used as the EOS symbol in the wait-k partitions, based on the model. "falcon" and "llama" are currenly supported. For example: "falcon" will use falcon\'s terminating symbol.')
    
    parser.add_argument('--custom-terminating-symbol', type=str,
                        help='Used solely for wait-k translations. Define a custom terminating symbol to be used as the EOS symbol in the wait-k partitions, will override terminating-symbol-type.')

    parser.add_argument('--translation-type', type=str, required=True,
                        help='Which translation to create, for example: "en-es" has direction en -> es')

    parser.add_argument('--must-c-dir', type=str, required=True,
                        help='The directory of the pre-downloaded MuST-C dataset where translations will be pulled from')

    parser.add_argument('--temporary-output-dir', type=str, default='smt_mustc_cleaned_jsons_temp/',
                        help='JSON files will be temporary created to upload the dataset, this directory is where they will be temporary stored')
    
    # parser.add_argument('--hf-local-cache-dir', type=str,
    #                     help='HuggingFace may save all datasets you upload to your local cache. If your current cache does not have the space for this, set a new cache directory here.')

    return parser


def main():

    parser = initialize_arg_parser()
    args = parser.parse_args()

    # =======================================
    # 0. Error check and establish variables
    # =======================================
    if len(args.translation_type.lower().split('-')) != 2:
        quit('\nTranslation type argument must be of format "languagecode1-languagecode2", such as "en-de", please try again.\n')

    source_lang_code, target_lang_code = args.translation_type.lower().split('-')
    if not language_pair_is_allowed(source_lang_code, target_lang_code):
        quit(f'\nLanguage pair "{args.translation_type}" is not currently supported, please look over which language pairs are supported.\n')

    if args.k_constants is None and args.f is None:
        quit(f'\nEither --k-constants or -f must be set to dictate what type of translation dataset to generate.\n')

    if args.k_constants is not None and (args.custom_terminating_symbol is None and args.terminating_symbol_type not in TERMINATING_SYMBOL_CODES):
        quit(f'\nIn order to generate wait-k datasets, describe a valid terminating-symbol-type or define a custom-terminating-symbol\n')

    # Establish our 2 main sets of filepaths
    source_filepaths = {
        'train': '',
        'tst-COMMON': '',   # Test 1 set for mustc
        'tst-HE': '',  # Test 2 set for mustc
        'dev': '',  # The validate set for mustc

        'precleaned-must-c-data': {  # Data that has already been cleaned
        }
    }
    target_filepaths = {
        'train': '',
        'tst-COMMON': '',
        'tst-HE': '',
        'dev': '', 

        'precleaned-must-c-data': {
        }
    }

    print('\nInput the account token for the HuggingFace account to upload a dataset to, this must be a WRITE token.')
    huggingface_hub.login()


    # =======================================
    # 1. Find filepaths of must-c data based on given must-c directory
    # =======================================
    for root, dirs, files in os.walk(args.must_c_dir):

        # Source
        populate_dictionary(root, files, source_filepaths, source_lang_code)

        # Target
        populate_dictionary(root, files, target_filepaths, target_lang_code)


    # =======================================
    # 2. Create cleaned MuST-C data files or regenerate them if that is necessary
    # =======================================
    if (source_filepaths['precleaned-must-c-data'] == {} or target_filepaths['precleaned-must-c-data'] == {} 
            or args.r):

        print(f'Creating cleaned MuST-C {source_lang_code} and {target_lang_code} data...')

        files_outputted = 0

        files_outputted += clean_mustc_data(source_lang_code, source_filepaths)
        files_outputted += clean_mustc_data(target_lang_code, target_filepaths)

        print(f'{files_outputted} cleaned files were created.')


    # =======================================
    # 3. Make train, test, and validation JSON files for wait-k and/or full-translation pairs
    # =======================================
    eos_token = ''
    if args.k_constants is not None:

        if args.custom_terminating_symbol is not None:
            eos_token = args.custom_terminating_symbol
        else:
            eos_token = TERMINATING_SYMBOL_CODES[args.terminating_symbol_type]
        
    create_translation_jsons(args.k_constants, args.temporary_output_dir, source_filepaths, 
                             target_filepaths, source_lang_code, target_lang_code, eos_token, args.f)
    

    # =======================================
    # 4. Push everything to the hub
    # =======================================
    os.environ['OPENBLAS_NUM_THREADS'] = '4'  # Fix a memory allocation bug
    datasets.disable_caching()

    # if args.hf_local_cache_dir is not None:  # Update local cache
    #     os.environ['HUGGINGFACE_HUB_CACHE'] = args.hf_local_cache_dir
    #     os.environ['TRANSFORMERS_CACHE'] = args.hf_local_cache_dir
    #     os.environ['HF_HOME'] = args.hf_local_cache_dir

    if args.k_constants is not None:

        for k_constant in args.k_constants:
            
            print(f'Attempting to upload the dataset file wait-{str(k_constant).zfill(2)}...')

            dataset_files = {
                'train': os.path.join(args.temporary_output_dir, f'wait{str(k_constant).zfill(2)}/must-c-{source_lang_code}-{target_lang_code}-train.json'),
                'test': os.path.join(args.temporary_output_dir, f'wait{str(k_constant).zfill(2)}/must-c-{source_lang_code}-{target_lang_code}-test.json'),
                'validation': os.path.join(args.temporary_output_dir, f'wait{str(k_constant).zfill(2)}/must-c-{source_lang_code}-{target_lang_code}-validation.json')
            }

            new_dataset = datasets.load_dataset('json', data_files=dataset_files)

            now = datetime.now()
            dataset_name = f'must-c-{source_lang_code}-{target_lang_code}-wait{str(k_constant).zfill(2)}_{now.hour}.{now.minute}'

            new_dataset.push_to_hub(dataset_name)

            print(f'Uploaded the wait-{str(k_constant).zfill(2)} dataset.')
    
    if args.f:
        print(f'Uploading the dataset file of whole translations...')

        dataset_files = {
            'train': os.path.join(args.temporary_output_dir, f'full/must-c-{source_lang_code}-{target_lang_code}-train.json'),
            'test': os.path.join(args.temporary_output_dir, f'full/must-c-{source_lang_code}-{target_lang_code}-test.json'),
            'validation': os.path.join(args.temporary_output_dir, f'full/must-c-{source_lang_code}-{target_lang_code}-validation.json')
        }

        new_dataset = datasets.load_dataset('json', data_files=dataset_files)

        now = datetime.now()
        dataset_name = f'must-c-{source_lang_code}-{target_lang_code}_{now.hour}.{now.minute}'

        new_dataset.push_to_hub(dataset_name)

        print(f'Uploaded the dataset of whole tranlsations.')

    datasets.enable_caching()

    # =======================================
    # 5. Delete our temporary json files
    # =======================================
    try:
        shutil.rmtree(args.temporary_output_dir)
    except Exception as e:
        print(f'Failed to delete temporary directory "{args.temporary_output_dir}", you will need to delete it yourself. Reason: {e}')


if __name__ == '__main__':
    main()
    