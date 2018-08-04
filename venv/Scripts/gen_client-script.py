#!C:\Users\jorge.melguizo\iCloudDrive\LaSalle\TFG\code\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'google-apitools==0.5.23','console_scripts','gen_client'
__requires__ = 'google-apitools==0.5.23'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('google-apitools==0.5.23', 'console_scripts', 'gen_client')()
    )
