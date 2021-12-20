import os
import json


def getTokenWithFile(filePath):
    statement = list()
    with open(filePath, 'r') as file:
        for line in file.readlines():
            items = line.split('\t')
            if len(items) > 0:
                statement.append(items[0])
            else:
                with open('./tmp/log.txt', 'a+') as logFile:
                    logFile.write(filePath)
                    logFile.write('\n')
                    logFile.write(line)
                    logFile.write('======')
                    logFile.write('\n')

    return statement


def readData(mode='train'):
    path = ''
    if mode == 'train':
        path = './data/train.jsonl'
    elif mode == 'test':
        path = './data/test.jsonl'
    else:
        path = './data/valid.jsonl'

    sampleList = list()
    with open(path, 'r') as file:
        for line in file.readlines():
            codeDict = json.loads(line)
            if codeDict is not None:
                sampleList.append(codeDict)

    print(len(sampleList))

    dataList = list()
    for index, codeDict in enumerate(sampleList):
        codeStr = codeDict.get('func', '')
        if len(codeStr) > 0:
            filePath = './tmp/code_{}.c'.format(index % 10)
            print(filePath)
            with open(filePath, 'w') as codeFile:
                codeFile.write(codeStr)

            tmpTokenFilePath = './tmp/tokens_{}.txt'.format(index % 10)
            os.system('clang -fsyntax-only -Xclang -dump-tokens {} >& {}'.format(filePath, tmpTokenFilePath))

            statement = getTokenWithFile(tmpTokenFilePath)
            if len(statement) > 0:
                dataList.append(statement)
                break

    print(dataList[0])

readData()