import re

def clean_en(input_string: str) -> str:

    # Apostraphe spacing fix
    input_string = re.sub(r" 's ", "'s ", input_string)
    input_string = re.sub(r" 's\.", "'s.", input_string)
    input_string = re.sub(r" 's!", "'s!", input_string)
    input_string = re.sub(r" 's\?", "'s?", input_string)
    input_string = re.sub(r" 's,", "'s,", input_string)
    input_string = re.sub(r" 's;", "'s;", input_string)
    input_string = re.sub(r" 's:", "'s:", input_string)

    # Improper em dash fix
    input_string = re.sub(r" — - ", ", ", input_string) 
    input_string = re.sub(r" — -", "", input_string)  
    input_string = re.sub(r"— - ", "", input_string) 

    # Removing "— (Laughter) —" -like occurances
    input_string = re.sub(r' — \(([^()\n]*)\) —', '', input_string)  
    input_string = re.sub(r' -- \(([^()\n]*)\) --', '', input_string)  
    input_string = re.sub(r' - \(([^()\n]*)\) -', '', input_string) 

    # Em dash spacing fix
    input_string = re.sub(r" — ", ", ", input_string) 
    input_string = re.sub(r" —", "", input_string)  
    input_string = re.sub(r"— ", "", input_string) 
    input_string = re.sub(r" -- ", ", ", input_string) 
    input_string = re.sub(r" --", "", input_string)  
    input_string = re.sub(r"-- ", "", input_string)  

    # General dash spacing fix
    input_string = re.sub(r'(\d) - (\d)', r'\1 to \2', input_string)  # '12 - 14' becomes '12 to 14'
    input_string = re.sub(r", - ", ", ", input_string) 
    input_string = re.sub(r" - - ", ", ", input_string) 
    input_string = re.sub(r"- - ", ", ", input_string) 
    input_string = re.sub(r" - ", ", ", input_string) 

    # Flag unclear lines
    # Lines with [unclear] will be marked with *** so that they 
    # will be discarded in the json formatting function
    input_string = re.sub(r"\[unclear\]", "[unclear***]", input_string, flags=re.I)

    # Remove everything in parentheses
    input_string = re.sub(r', \(([^()\n]*)\),', ',', input_string)  # Deal with commas first: ", (Laughter)," --> ","
    input_string = re.sub(r' \(([^()\n]*)\)', '', input_string)  # Then, left-spaces
    input_string = re.sub(r'\(([^()\n]*)\) ', '', input_string)  # Then, right-spaces
    input_string = re.sub(r'\(([^()\n]*)\)', '', input_string)  # Then, any that are left over

    # Remove brackets, are unnecessary in english versions
    input_string = re.sub(r'\[', '', input_string)
    input_string = re.sub(r'\]', '', input_string)

    # Remove music symbols
    input_string = re.sub(r' ♫', '', input_string)
    input_string = re.sub(r'♫ ', '', input_string)
    input_string = re.sub(r'♫', '', input_string)

    # Improper punctuation spacing fix
    input_string = re.sub(r' / ', '/', input_string)
    input_string = re.sub(r'# ', '#', input_string)

    # Miscellaneous
    # input_string = re.sub(r' ²', '²', input_string)
    # input_string = re.sub(r' \+ \+', '++', input_string)
    # input_string = re.sub(r'A \+', 'A+', input_string)
    
    return input_string


def clean_es(input_string: str) -> str:

    # Apostraphe spacing fix
    input_string = re.sub(r" 's ", "'s ", input_string)
    input_string = re.sub(r" 's\.", "'s.", input_string)
    input_string = re.sub(r" 's!", "'s!", input_string)
    input_string = re.sub(r" 's\?", "'s?", input_string)
    input_string = re.sub(r" 's,", "'s,", input_string)
    input_string = re.sub(r" 's;", "'s;", input_string)
    input_string = re.sub(r" 's:", "'s:", input_string)

    # Removing "— (Risas) —" -like occurances
    input_string = re.sub(r' — \(([^()\n]*)\) —', '', input_string)  
    input_string = re.sub(r' -- \(([^()\n]*)\) --', '', input_string)  
    input_string = re.sub(r' - \(([^()\n]*)\) -', '', input_string)  

    # General dash spacing fix
    input_string = re.sub(r'(\d) - (\d)', r'\1 a \2', input_string)  # '12 - 14' becomes '12 a 14'
    input_string = re.sub(r", - ", ", ", input_string) 
    input_string = re.sub(r" - - ", ", ", input_string) 
    input_string = re.sub(r"- - ", ", ", input_string) 
    input_string = re.sub(r" - ", ", ", input_string) 

    # Em dash spacing fix
    input_string = re.sub(r" — ", ", ", input_string) 
    input_string = re.sub(r" —", "", input_string)  
    input_string = re.sub(r"— ", "", input_string)  

    

    # Flag unclear lines
    # Lines with "poco claro" will be marked with *** so that they 
    # will be discarded in the json formatting function
    input_string = re.sub(r"poco claro", "poco claro***", input_string, flags=re.I)
    input_string = re.sub(r"\[poco claro\]", "[poco claro]***", input_string, flags=re.I)

    # Fix improper front-tick use
    input_string = re.sub(r' ´ s([ !\?\.,])', '\'s\\1', input_string)  # Sometimes front-tick is meant as an apostraphe
    input_string = re.sub(r'´ ', '', input_string)
    input_string = re.sub(r' ´', '', input_string)

    # Remove all words inside square brackets, are unnecessary in Spanish versions
    input_string = re.sub(r' \[([^\[\]\n]*)\]', '', input_string)  # Then, left-spaces
    input_string = re.sub(r'\[([^\[\]\n]*)\] ', '', input_string)  # Then, right-spaces
    input_string = re.sub(r'\[([^\[\]\n]*)\]', '', input_string)  # Then, any that are left over

    # Remove everything in parentheses
    input_string = re.sub(r', \(([^()\n]*)\),', ',', input_string)  # Deal with commas first: ", (Laughter)," --> ","
    input_string = re.sub(r' \(([^()\n]*)\)', '', input_string)  # Then, left-spaces
    input_string = re.sub(r'\(([^()\n]*)\) ', '', input_string)  # Then, right-spaces
    input_string = re.sub(r'\(([^()\n]*)\)', '', input_string)  # Then, any that are left over

    # Remove music symbols
    input_string = re.sub(r' ♫', '', input_string)
    input_string = re.sub(r'♫ ', '', input_string)
    input_string = re.sub(r'♫', '', input_string)

    # Improper punctuation spacing fix
    input_string = re.sub(r'# ', '#', input_string)

    # Miscellaneous
    # input_string = re.sub(r' ²', '²', input_string)
    # input_string = re.sub(r' \+ \+', '++', input_string)
    # input_string = re.sub(r'A \+', 'A+', input_string)

    return input_string


def clean_de(input_string: str) -> str:

    # Apostraphe spacing fix
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

    # Em dash spacing fix
    input_string = re.sub(r" — ", ", ", input_string) 
    input_string = re.sub(r" —", "", input_string)  
    input_string = re.sub(r"— ", "", input_string) 
    input_string = re.sub(r" -- ", ", ", input_string) 
    input_string = re.sub(r" --", "", input_string)  
    input_string = re.sub(r"-- ", "", input_string)  

    # General dash spacing fix
    input_string = re.sub(r'(\d) - (\d)', r'\1 a \2', input_string)  # '12 - 14' becomes '12 a 14'
    input_string = re.sub(r", - ", ", ", input_string) 
    input_string = re.sub(r" - - ", ", ", input_string) 
    input_string = re.sub(r"- - ", ", ", input_string) 
    input_string = re.sub(r" - ", ", ", input_string) 

    # Flag unclear lines
    # Lines with "unklar" will be marked with *** so that they 
    # will be discarded in the json formatting function
    input_string = re.sub(r"\[unklar\]", "[unklar]***", input_string, flags=re.I)
    input_string = re.sub(r"unklar", "unklar***", input_string, flags=re.I)

    # Fix improper front-tick use
    input_string = re.sub(r' ´ s([ !\?\.,])', '\'s\\1', input_string)  # Sometimes front-tick is meant as an apostraphe
    input_string = re.sub(r'´ ', '', input_string)
    input_string = re.sub(r' ´', '', input_string)

    # Remove all words inside square brackets, are unnecessary in German versions
    input_string = re.sub(r' \[([^\[\]\n]*)\]', '', input_string)  # Then, left-spaces
    input_string = re.sub(r'\[([^\[\]\n]*)\] ', '', input_string)  # Then, right-spaces
    input_string = re.sub(r'\[([^\[\]\n]*)\]', '', input_string)  # Then, any that are left over

    # Remove everything in parentheses
    input_string = re.sub(r', \(([^()\n]*)\),', ',', input_string)  # Deal with commas first: ", (Laughter)," --> ","
    input_string = re.sub(r' \(([^()\n]*)\)', '', input_string)  # Then, left-spaces
    input_string = re.sub(r'\(([^()\n]*)\) ', '', input_string)  # Then, right-spaces
    input_string = re.sub(r'\(([^()\n]*)\)', '', input_string)  # Then, any that are left over

    # Remove music symbols
    input_string = re.sub(r' ♫', '', input_string)
    input_string = re.sub(r'♫ ', '', input_string)
    input_string = re.sub(r'♫', '', input_string)

    # Improper punctuation spacing fix
    input_string = re.sub(r'# ', '#', input_string)

    # Miscellaneous
    # input_string = re.sub(r' ²', '²', input_string)
    # input_string = re.sub(r' \+ \+', '++', input_string)
    # input_string = re.sub(r'A \+', 'A+', input_string)

    return input_string


def clean_fr(input_string: str) -> str:

    # Doube quotes spacing fix
    input_string = re.sub(r" \"\" ", " \"", input_string) 

    # Improper em dash fix
    input_string = re.sub(r", — ", ", ", input_string) 
    input_string = re.sub(r" — - ", " — ", input_string) 
    input_string = re.sub(r" — -", " —", input_string)  
    input_string = re.sub(r"— - ", "— ", input_string) 

    # Removing "— (Rire) —" -like occurances
    input_string = re.sub(r' — \(([^()\n]*)\) —', '', input_string)  
    input_string = re.sub(r' -- \(([^()\n]*)\) --', '', input_string)  
    input_string = re.sub(r' - \(([^()\n]*)\) -', '', input_string) 

    # Em dash spacing fix
    input_string = re.sub(r" — ", ", ", input_string) 
    input_string = re.sub(r" —", "", input_string)  
    input_string = re.sub(r"— ", "", input_string) 

    # General dash spacing fix
    input_string = re.sub(r", - ", ", ", input_string) 
    input_string = re.sub(r" - ", " — ", input_string) 
    input_string = re.sub(r" - - ", ", ", input_string) 
    input_string = re.sub(r"- - ", ", ", input_string) 

    # Flag unclear lines
    # Lines with "pas clair" will be marked with *** so that they   
    # will be discarded in the json formatting function
    input_string = re.sub(r"\[pas clair\]", "[pas clair]***", input_string, flags=re.I)

    # Fix improper front-tick use
    input_string = re.sub(r' ´ ', '\'', input_string)  # Front-ticks are used nearly always as apostraphes

    # Remove all words inside square brackets, are unnecessary in French versions
    input_string = re.sub(r' \[([^\[\]\n]*)\]', '', input_string)  # First, brackets with spaces on the left
    input_string = re.sub(r'\[([^\[\]\n]*)\] ', '', input_string)  # Then, brackets with spaces on the right
    input_string = re.sub(r'\[([^\[\]\n]*)\]', '', input_string)  # Any that are left over

    # Remove everything in parentheses
    input_string = re.sub(r', \(([^()\n]*)\),', ',', input_string)  # Deal with commas first: ", (Laughter)," --> ","
    input_string = re.sub(r' \(([^()\n]*)\)', '', input_string)  # Then, left-spaces
    input_string = re.sub(r'\(([^()\n]*)\) ', '', input_string)  # Then, right-spaces
    input_string = re.sub(r'\(([^()\n]*)\)', '', input_string)  # Then, any that are left over

    # Remove music symbols
    input_string = re.sub(r' ♫', '', input_string)
    input_string = re.sub(r'♫ ', '', input_string)
    input_string = re.sub(r'♫', '', input_string)

    # Improper punctuation spacing fix
    input_string = re.sub(r'# ', '#', input_string)
    # input_string = re.sub(r' ²', '²', input_string)
    # input_string = re.sub(r' \+ \+', '++', input_string)
    # input_string = re.sub(r'A \+', 'A+', input_string)

    return input_string
