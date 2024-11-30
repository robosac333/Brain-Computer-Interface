import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/abhishek/Brain-Computer-Interface/bci_py_ws/install/bci_package'
