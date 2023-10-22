# FORMATTING SCRIPT FOR MUST-C EN-ES


# This is to check for:
    # Misuse of " 's"
    # Check for em-dashes
    # Check for " punctuation
    # Check for ... punctuation
    # Check for [
    # Check for (
    # Check for "♫"
    # Check for "`"
    # Any punctuation in general

# To-deletes--delete any lines that include these:
    # "^"
    # "***" (this is my flag for unclear words)
    # (empty lines)
    # "_" (the datasets are weird about this character)
    
import re

EN_INPUT_OUTPUT = [
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/train/txt/train.en', '/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/train/txt/train-02.en'),
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-COMMON/txt/tst-COMMON.en', '/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-COMMON/txt/tst-COMMON-02.en'),
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-HE/txt/tst-HE.en','/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-HE/txt/tst-HE-02.en'),
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/dev/txt/dev.en','/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/dev/txt/dev-02.en')
]

ES_INPUT_OUTPUT = [
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/train/txt/train.es', '/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/train/txt/train-02.es'),
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-COMMON/txt/tst-COMMON.es', '/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-COMMON/txt/tst-COMMON-02.es'),
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-HE/txt/tst-HE.es', '/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/tst-HE/txt/tst-HE-02.es'),
    ('/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/dev/txt/dev.es', '/nfs/hpc/share/wildma/dataset_prep/MuST-C-en-es/data/dev/txt/dev-02.es')
]

def fix_en(input_string: str):
    # APOSTRAPHE SPACING FIX
    input_string = re.sub(r" 's ", "'s ", input_string)
    input_string = re.sub(r" 's\.", "'s.", input_string)
    input_string = re.sub(r" 's!", "'s!", input_string)
    input_string = re.sub(r" 's\?", "'s?", input_string)
    input_string = re.sub(r" 's,", "'s,", input_string)
    input_string = re.sub(r" 's;", "'s;", input_string)
    input_string = re.sub(r" 's:", "'s:", input_string)

    # Removing "— (Laughter) —" -like occurances
    input_string = re.sub(r' — \(([^()\n]*)\) —', '', input_string)  
    input_string = re.sub(r' -- \(([^()\n]*)\) --', '', input_string)  
    input_string = re.sub(r' - \(([^()\n]*)\) -', '', input_string) 

    # EM-DASH FIX
    input_string = re.sub(r" — ", ", ", input_string) 
    input_string = re.sub(r" —", "", input_string)  
    input_string = re.sub(r"— ", "", input_string)  

    # General dash fix
    input_string = re.sub(r'(\d) - (\d)', r'\1 to \2', input_string)  # '12 - 14' becomes '12 to 14'
    input_string = re.sub(r", - ", ", ", input_string) 
    input_string = re.sub(r" - - ", ", ", input_string) 
    input_string = re.sub(r"- - ", ", ", input_string) 
    input_string = re.sub(r" - ", ", ", input_string) 

    # FLAG UNCLEAR LINES
    # Lines with [unclear] will be marked with *** so that they 
    # will be discarded in the json formatting function
    input_string = re.sub(r"\[unclear\]", "[unclear***]", input_string, flags=re.I)

    # PARENTHESES FIX
    # input_string = re.sub(r', \([^)]*\),', ',', input_string)  # This old regex code was invalid because it should not delete any newlines
    input_string = re.sub(r', \(([^()\n]*)\),', ',', input_string)  # Deal with commas first: ", (Laughter)," --> ","
    input_string = re.sub(r' \(([^()\n]*)\)', '', input_string)  # Then, left-spaces
    input_string = re.sub(r'\(([^()\n]*)\) ', '', input_string)  # Then, right-spaces
    input_string = re.sub(r'\(([^()\n]*)\)', '', input_string)  # Then, any that are left over
    # ", ()" sometimes messes up sentence flow, but not often, so I don't think there's a need to remove it

    # REMOVE UNNECESSARY BRACKETS
    input_string = re.sub(r'\[', '', input_string)
    input_string = re.sub(r'\]', '', input_string)

    # MUSIC SYMBOLS
    input_string = re.sub(r' ♫', '', input_string)
    input_string = re.sub(r'♫ ', '', input_string)
    input_string = re.sub(r'♫', '', input_string)

    # Miscellaneous Silly stuff
    input_string = re.sub(r' ²', '²', input_string)
    input_string = re.sub(r' \+ \+', '++', input_string)
    input_string = re.sub(r'A \+', 'A+', input_string)
    
    # Fix improper hashtag use
    input_string = re.sub(r'# ', '#', input_string)

    # I think the use of ":" is usable here
    return input_string


def fix_es(input_string: str):
    # APOSTRAPHE SPACING FIX
    input_string = re.sub(r" 's ", "'s ", input_string)
    input_string = re.sub(r" 's\.", "'s.", input_string)
    input_string = re.sub(r" 's!", "'s!", input_string)
    input_string = re.sub(r" 's\?", "'s?", input_string)
    input_string = re.sub(r" 's,", "'s,", input_string)
    input_string = re.sub(r" 's;", "'s;", input_string)
    input_string = re.sub(r" 's:", "'s:", input_string)

    # Removing "— (Laughter) —" -like occurances
    input_string = re.sub(r' — \(([^()\n]*)\) —', '', input_string)  
    input_string = re.sub(r' -- \(([^()\n]*)\) --', '', input_string)  
    input_string = re.sub(r' - \(([^()\n]*)\) -', '', input_string)  

    # General dash fix
    input_string = re.sub(r'(\d) - (\d)', r'\1 a \2', input_string)  # '12 - 14' becomes '12 a 14'
    input_string = re.sub(r", - ", ", ", input_string) 
    input_string = re.sub(r" - - ", ", ", input_string) 
    input_string = re.sub(r"- - ", ", ", input_string) 
    input_string = re.sub(r" - ", ", ", input_string) 

    # EM-DASH FIX
    input_string = re.sub(r" — ", ", ", input_string) 
    input_string = re.sub(r" —", "", input_string)  
    input_string = re.sub(r"— ", "", input_string)  

    # FLAG UNCLEAR LINES
    # Lines with "poco claro" will be marked with *** so that they 
    # will be discarded in the json formatting function
    input_string = re.sub(r"poco claro", "poco claro***", input_string, flags=re.I)
    input_string = re.sub(r"\[poco claro\]", "[poco claro]***", input_string, flags=re.I)

    # Fix weird front-tick use
    input_string = re.sub(r' ´ s([ !\?\.,])', '\'s\\1', input_string)  # Sometimes front-tick is meant as an apostraphe
    input_string = re.sub(r'´ ', '', input_string)
    input_string = re.sub(r' ´', '', input_string)

    # Remove words inside all square brackets
    # input_string = re.sub(r' \[[^)]*\]', '', input_string)  # This old regex code was invalid because it should not delete any newlines
    input_string = re.sub(r' \[([^\[\]\n]*)\]', '', input_string)  # Then, left-spaces
    input_string = re.sub(r'\[([^\[\]\n]*)\] ', '', input_string)  # Then, right-spaces
    input_string = re.sub(r'\[([^\[\]\n]*)\]', '', input_string)  # Then, any that are left over

    # PARENTHESES FIX
    input_string = re.sub(r', \(([^()\n]*)\),', ',', input_string)  # Deal with commas first: ", (Laughter)," --> ","
    input_string = re.sub(r' \(([^()\n]*)\)', '', input_string)  # Then, left-spaces
    input_string = re.sub(r'\(([^()\n]*)\) ', '', input_string)  # Then, right-spaces
    input_string = re.sub(r'\(([^()\n]*)\)', '', input_string)  # Then, any that are left over
    # ", ()" sometimes messes up sentence flow, but not often, so I don't think there's a need to remove it

    # MUSIC SYMBOLS
    input_string = re.sub(r' ♫', '', input_string)
    input_string = re.sub(r'♫ ', '', input_string)
    input_string = re.sub(r'♫', '', input_string)

    # Fix improper hashtag use
    input_string = re.sub(r'# ', '#', input_string)

    # Miscellaneous Silly stuff
    input_string = re.sub(r' ²', '²', input_string)
    input_string = re.sub(r' \+ \+', '++', input_string)
    input_string = re.sub(r'A \+', 'A+', input_string)

    return input_string


def main():

    files_outputted = 0

    for file_pair in EN_INPUT_OUTPUT:

        src_data = ''

        with open(file_pair[0], 'r') as f:
            src_data = f.read()

        src_data = fix_en(src_data)

        with open(file_pair[1], 'w') as f:
            f.write(src_data)
            files_outputted += 1
            print(f'Wrote file {files_outputted}')


    for file_pair in ES_INPUT_OUTPUT:

        src_data = ''

        with open(file_pair[0], 'r') as f:
            src_data = f.read()

        src_data = fix_es(src_data)  # The different function is important

        with open(file_pair[1], 'w') as f:
            f.write(src_data)
            files_outputted += 1
            print(f'Wrote file {files_outputted}')

    print('Finished writing to all files.')


if __name__ == "__main__":
    main()
