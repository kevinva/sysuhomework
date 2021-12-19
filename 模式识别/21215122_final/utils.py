import os
import json

def readData(mode='train'):
    path = ''
    if mode == 'train':
        path = './data/train.jsonl'
    elif mode == 'test':
        path = './data/test.jsonl'
    else:
        path = './data/valid.jsonl'

    resultList = list()
    with open(path, 'r') as file:
        for line in file.readlines():
            codeDict = json.loads(line)
            if codeDict is not None:
                resultList.append(codeDict)

    print(len(resultList))

    for index, codeDict in enumerate(resultList):
        codeStr = codeDict.get('func', '')
        if len(codeStr) > 0:
            if index % 1000 == 0:
                filePath = './data/output/code_{}.c'.format(index)
                print(filePath)
                with open(filePath, 'w') as codeFile:
                    codeFile.write(codeStr)

                os.system('clang -fsyntax-only -Xclang -dump-tokens {} >& ./tmp/tmp_{}.txt'.format(filePath, index))

                

readData()


