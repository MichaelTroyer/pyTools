# -*- coding: utf-8 -*-
"""
Tools for working with string data types.

Particularly useful for enforcing bytes or unicode string types in 
Python2 and Python 3.

Based on Effective Python (Item 3) by Brett Slatkin.
"""

import sys


if (sys.version_info > (3, 0)):
    ### Python 3 ###

    def to_str(bytes_or_str):
        """
        Take a bytes string or a unicode string and always return a unicode string.
        In Python 3 <str> is Unicode and bytes have to be declared (b'...').
        NOTE: Default encoding in Python 3 is UTF-8.
        """
        if isinstance(bytes_or_str, bytes):
            # A bytes string - decode to unicode
            value = bytes_or_str.decode('utf-8')
        else:
            # Already unicode
            value = bytes_or_str
        return value
    
    def to_bytes(bytes_or_str):
        """
        Take a bytes string or a unicode string and always return a bytes string.
        In Python 3 <str> is Unicode and bytes have to be declared (b'...').
        NOTE: Default encoding in Python 3 is UTF-8.
        """
        if isinstance(bytes_or_str, str):
            # Unicode - encode to bytes
            value = bytes_or_str.encode('utf-8')
        else:
            # Already bytes
            value = bytes_or_str
        return value 

else:
    ### Python 2 ###
    def to_unicode(unicode_or_str):
        """
        Take a bytes string or a unicode string and always return a unicode string.
        In Python 2 <str> is bytes and unicode has to be declared (u'...').
        NOTE: Default encoding in Python 2 is ASCII.
        """
        if isinstance(unicode_or_str, str):
            # A bytes string - decode to unicode
            value = unicode_or_str.decode('utf-8')
        else:
            # Already unicode
            value = unicode_or_str
        return value

    def to_str(unicode_or_str):
        """
        Take a bytes string or a unicode string and always return a bytes string.
        In Python 2 <str> is bytes and unicode has to be declared (u'...').
        NOTE: Default encoding in Python 2 is ASCII.
        """
        if isinstance(unicode_or_str, unicode):
            # A unicode string - encode to bytes
            value = unicode_or_str.encode('utf-8')
        else:
            # Already bytes
            value = unicode_or_str
        return value