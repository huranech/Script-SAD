import imblearn
import sklearn
import pandas as pd
import pickle
import sys
import numpy
import getopt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'p:m:f:h',['path=','model=','testFile=','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-p','--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-m', '--model'):
            m = arg
        elif opt in ('-h','--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName -f testFileName\n ')
            exit(1)

    
    if p == './':
        model=p+str(m)
        iFile = p+ str(f)
    else:
        model=p+"/"+str(m)
        iFile = p+"/" + str(f)