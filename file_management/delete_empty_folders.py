# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 08:29:16 2018

@author: mtroyer
"""
import os
import shutil


def delete_empty_folders(path):
    for folder in os.listdir(path):
        full_path = os.path.join(path, folder)
        if not os.listdir(full_path):  # is empty
            shutil.rmtree(full_path)