#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:
Michael D. Troyer

"""
def print_dict_to_table(data_dict, name=None, pad=3):
    max_k_len = max([len(str(k)) for k in data_dict.keys()])
    max_v_len = max([len(str(v)) for v in data_dict.values()])
    k_width = max_k_len+2*pad
    v_width = max_v_len+2*pad
    name = name if name else 'Dictionary'
    print
    print name.center(k_width+v_width+3, '=')

    for k, v in data_dict.items():
        print '|'+str(k).center(k_width)+'|'+str(v).center(v_width)+'|'
