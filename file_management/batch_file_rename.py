import os
import re

import PySimpleGUI as sg


class BatchRenameFiles(object):

    def __init__(self, testMode=False, testInputs=None):
        if testMode: 
            self.event, self.inputs = 'testing', testInputs
        else:
            self.event, self.inputs = self.getInputs()
        # print(self.inputs)
        self.findText = self.inputs['findText']
        self.ignoreCase = self.inputs['ignoreCase']
        self.regex= self.inputs['regex']
        self.replaceText = self.inputs['replaceText']
        self.directory = self.inputs['directory']
        self.recurse = self.inputs['recurse']
        self.prefix = self.inputs['prefix']
        self.suffix = self.inputs['suffix']

        #TODO: screen prefix and suffix simulataneous input
        if self.regex:
            if not self.validRegexString(self.findText):
                raise ValueError('Input is not a valid regular expression')

        self.renamedFiles = self.main()

    @staticmethod
    def getInputs():
        layout = [      
            [sg.Text('File name text to replace', size=(25, 1)),
                sg.Checkbox('Ignore case', key='ignoreCase'),
                sg.Checkbox('Regex', key='regex')],  
            [sg.InputText(key='findText')],
            [sg.Text('Replacement text', size=(25, 1)),
                sg.Checkbox('Prefix', key='prefix'),
                sg.Checkbox('Suffix', key='suffix')],
            [sg.InputText(key='replaceText')],
            [sg.Text('Directory of files to be renamed', size=(25, 1)), 
                sg.Checkbox('Recurse sub-directories', key='recurse')],
            [sg.InputText(key='directory'), sg.FolderBrowse()],
            [sg.Submit(), sg.Cancel()],
            ] 

        window = sg.Window(
            'Batch File Rename',
            default_element_size=(60, 1),
            grab_anywhere=False).Layout(layout)      

        event, values = window.Read()
        if event in (None, 'Cancel'):
            raise Exception('Operation cancelled by user')

        return event, values

    @staticmethod
    def validRegexString(regexString):
        try:
            re.compile(regexString)
            return True
        except re.error:
            return False

    def getNewFileName(self, fileName):
        if self.prefix:
            #TODO: only prefix/suffix on findText match
            #TODO: leave blank to prefix/suffix all
            return self.replaceText + fileName
        elif self.suffix:
            name, ext = os.path.splitext(fileName)
            return name + self.replaceText + ext
        else:
            name, ext = os.path.splitext(fileName)
            ignore = re.IGNORECASE if self.ignoreCase else 0
            return re.sub(self.findText, self.replaceText, name, flags=ignore) + ext
        
    def renameFile(self, dir, fileName):
        new_f = self.getNewFileName(fileName)
        if fileName != new_f:
            os.rename(os.path.join(dir, fileName), os.path.join(dir, new_f))
            return new_f
        return None

    @staticmethod
    def returnResults(results):
        layout = [      
            [sg.Text(
                'Results: renamed {} files:\n{}'.format(
                    len(results),
                    '\n'.join(['[{}] -->\t[{}]'.format(k, v) for k, v in results.items()]
                        )))],
            [sg.Ok()],
            ] 

        window = sg.Window('Results', grab_anywhere=False).Layout(layout)      
        event, values = window.Read()
        return event, values

    def main(self):

        renamedFiles = {}

        if self.recurse:
            for root, dirs, files in os.walk(self.directory):
                for f in files:
                    res = self.renameFile(root, f)
                    if res: renamedFiles[f] = res
        else:
            for f in os.listdir(self.directory):
                res = self.renameFile(self.directory, f)
                if res: renamedFiles[f] = res

        self.returnResults(renamedFiles)

        return renamedFiles

if __name__ == '__main__':
    renamed = BatchRenameFiles()

    # print(renamed.renamedFiles)