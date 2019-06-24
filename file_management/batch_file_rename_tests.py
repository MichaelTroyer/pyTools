from batch_file_rename import BatchRenameFiles


def test_validateRegexString(test_cases):
    for test, result in test_cases.items():
        if not BatchRenameFiles.validRegexString(test) == result:
            raise Exception('validateRegexString failed', test, result)


def test_getNewFilename(test_batch, test_cases):
    for test, result in test_cases.items():
        if not test_batch.getNewFileName(test) == result:
            raise Exception('getNewFilename failed', test, result)


if __name__ == '__main__':

    validateRegexStringTestCases = {
    # String: True/False,
    'Test[0-9]{2}String': True,
    ']': True,
    }
    test_validateRegexString(validateRegexStringTestCases)



    param_values = {
        'findText' : '',
        'ignoreCase' : '',
        'regex' : '',
        'replaceText' : '',
        'directory' : '',
        'recurse' : '',
        'prefix' : '',
        'suffix' : '',
        }
        
    test_cases = [
        # oldName: newName,
    ]


    # test_batch = BatchRenameFiles(testMode=True, testValues)
    # test_getNewFilename(test_batch, getNewFilenameTestCases)

    