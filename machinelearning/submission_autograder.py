#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))

