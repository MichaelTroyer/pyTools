# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:49:05 2018

@author: mtroyer
"""


import os
import sqlite3
import pandas as pd


class WrongDatabaseType(Exception):
    pass


def read_sqlite(runtime_gdb_path, output_path, filter_system_tables=True):
    
    if not runtime_gdb_path.endswith('.geodatabase'):
        raise WrongDatabaseType
        
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    system_tables = ['st_', 'GDB_', 'sqlite_']

    try:
        db = sqlite3.connect(runtime_gdb_path)
        cursor = db.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        
        for table in tables:
            try:
                table_name = table[0]
                
                # Skip system tables
                if filter_system_tables:
                    if any([table_name.startswith(st) for st in system_tables]):
                        continue
                    
                # Handle attachments
                if table_name.endswith('__ATTACH'):
                    attachments_path = os.path.join(output_path, 'attachments')
                    if not os.path.exists(attachments_path):
                        os.mkdir(attachments_path)
                    
                    cursor.execute("SELECT DATA, ATT_NAME FROM {}".format(table_name))
                    for ix, row in enumerate(cursor.fetchall()):
                        attachment, filename = row[0], row[1]
                        if attachment:
                            with open(os.path.join(attachments_path, '{}_{}'.format(ix, filename)), 'wb') as f:
                                f.write(attachment)
                
                df = pd.read_sql_query("SELECT * from {}".format(table_name), db)
                if not df.empty:                            
                    df.to_csv(os.path.join(output_path, table_name + '.csv'), index_label='index')
            except Exception as err:
                print '[+] Error reading table: {}'.format(err)
    except Exception as err:
        print '[+] Error connecting to database: {}'.format(err)
