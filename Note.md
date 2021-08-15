# Cleaner Codes 

- Moduler Codes: 
    - Split into functions and modules (files)
    - "DRY" (Dont Repeat Yourself) concept 
    - This will reduece the repetation and improve the readability
    - Do not modularie too much. 

- Naming
    - descriptive 
    - type ("is_minor", "age_list")
    - short 
    - arbitrary variables names
         - argument => 'arr'
         - 'n'  

- Documentation
    - Inine Comments - line level
    - DocStrings - for function and module (all functions should have a docstring). Args, Returns
    - ReadMe file - project level  

- Refactoring 
    - Restructuring code to improve after writing a code 
    - Speed up the development speed in the long run
    - Allows to reuse more

- Efficient Code 
    - Reducing run time 
    - Reducing space in memory
    - Context independent

- To speed up 
    - use vector operations (numpy, pandas)
    - use Sets (to find the common elecments in lists)
    - use time library to see the time. 
    - use numpy indexing instead of evaluate each element


- Use whitespace (white lines, indentation)
- One line < 79 chars (PEP8 guideline)
- Tools to follow PEP8 standards 
    - autopep8 (install from pip)  -> "auto pep8 --in-place --aggressive --aggressive common_books.py" 
        - automatically change the file
    - pylint (install from pip)  - "pylint common_books.py" 
        - scores the file 


# Production-Ready Code 

- try ~ except statement 
     - Good practice is expecting specific type of error (ex: 'FileNotFoundError'. This can be tuple)
     - "assert" statement is useful. Use "AssertionError" for error handling.
- Testing
    - ML model testing is different from traditional software testing. 
    - lack of testing is a big complaints from software engineers to data scientists 
    - "Unit Tests" 
        -  independent code to test a funtion multiple times 
        - 'assert' to check the answers are correct
    - Test Tool - 
        - "pytest": pip install pytest. 
        - just run "pytest" 
        - test file name should start with "test_"
    - "Test Deriven Development"  - write tests before writing the codes 
        - standard practice of software engineering 

- Logging 
    - "logging" library of Python - instead of using "print". No need a human to monitor print message and makes it possible to fix the issues later, seeing log files. 
    - "logging.basicConfig" 
        - filename
        - level (INFO/WARNING/ERROR)
        - filemode (W)
        - format - name/levelname/message
    - "logging.info('Success') " 
    - "logging.error" 

- Model Drift 
    - Model may not perform for new data
    - Metrics: clicks etc. will be decline 
    - May need to retrain - automated or non-automated 
        - each week/month/year
        - automated - simple retraning  
        - non-automated - complicated (new features, new architectures)