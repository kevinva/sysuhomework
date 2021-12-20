import os
import json
import time

FUNCTION_ID_PREFIX = 'hoho_func'
VAR_ID_PREFIX = 'hoho_var'
POINTER_TYPE_ID_PREFIX = 'hoho_pointer_type'
VAR_TYPE_ID_PREFIX = 'hoho_var_type'
STRING_CONSTANT_PREFIX = 'hoho_str_constant'
NUMERIC_CONSTANT_PREFIX = 'hoho_numeric_constant'
C_VAR_TYPE_LIST = ['char', 'wchar_t', 'short', 'int', 'int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t', 
                   'uint16_t', 'uint32_t', 'uint64_t', 'float', 'double', 'long', 'size_t', 'bool'] 


def getTokenWithFile(filePath):
    tokenList = list()
    wrongFilePath = './tmp/wrong_{}.txt'.format(time.time())
    with open(filePath, 'r') as file:
        lineList = file.readlines()
        isLastType = False
        for index, line in enumerate(lineList):
            items = line.split('\t')
            if len(items) > 0:
                subItems = items[0].split(' ')
                if len(subItems) > 1:
                    val = subItems[1][1:][:-1]   # [1:]为去掉前单引号，[:-1]为去掉后单引号

                    if subItems[0] == 'identifier':
                        if isLastType:
                            isLastType = False
                            tokenList.append('{}:{}'.format(VAR_ID_PREFIX, subItems[1]))
                            continue

                        if index + 1 < len(lineList):
                            nextItem = line[index + 1]
                            nextSubItems = nextItem.split(' ')
                            if len(nextSubItems) > 1:
                                if nextSubItems[0] == 'l_paren':
                                    # tokenList.append('{}:{}'.format(FUNCTION_ID_PREFIX, subItems[1]))  # 自定义函数名
                                    tokenList.append('{}'.format(FUNCTION_ID_PREFIX))  # 自定义函数名
                                elif nextSubItems[0] == 'star':
                                    # tokenList.append('{}:{}'.format(POINTER_TYPE_ID_PREFIX, subItems[1]))  # 指针类型
                                    tokenList.append('{}'.format(POINTER_TYPE_ID_PREFIX))  # 指针类型
                                elif nextSubItems[0] == 'identifier':
                                    if val in C_VAR_TYPE_LIST:
                                        tokenList.append(val)
                                    else:
                                        # tokenList.append('{}:{}'.format(VAR_TYPE_ID_PREFIX, subItems[1]))  # 用户自定义变量类型
                                        tokenList.append('{}'.format(VAR_TYPE_ID_PREFIX))  # 用户自定义变量类型
                                    isLastType = True
                                else:
                                    tokenList.append(val)   
                            else:
                                tokenList.append(val)   
                        else:
                            tokenList.append(val) 
                    else:
                        tokenList.append(val)
   
                else:
                    with open(wrongFilePath, 'w') as wrongFile:
                        wrongFile.writelines(lineList)
                    continue
            else:
                with open(wrongFilePath, 'w') as wrongFile:
                    wrongFile.writelines(lineList)
                continue

    return tokenList


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
            print(filePath)   ## hoho_debug
            with open(filePath, 'w') as codeFile:
                codeFile.write(codeStr)

            tmpTokenFilePath = './tmp/tokens_{}.txt'.format(index % 10)
            os.system('clang -fsyntax-only -Xclang -dump-tokens {} >& {}'.format(filePath, tmpTokenFilePath))

            statement = getTokenWithFile(tmpTokenFilePath)
            if len(statement) > 0:
                dataList.append(statement)
                break   ## hoho_debug

    print(dataList[0])

readData()