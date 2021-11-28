import math

# 找两条线的交点
def findCrossPointWithTwoLine(line1, line2):
    x11, y11, x21, y21 = line1
    x12, y12, x22, y22 = line2 
    k1 = (y21 - y11) / (x21 - x11)
    k2 = (y22 - y12) / (x22 - x12)
    b1 = y11 - k1 * x11
    b2 = y12 - k2 * x12

    assert k1 != k2
    
    crossX = (b2 - b1) / (k1 - k2)
    crossY = k1 * crossX + b1
    return int(crossX), int(crossY)  # 必须返回整型


# 找四边形4个角点，找上、左、下、右4条边，
def find4CrossPointWithLines(lines):
    pEdges = list() 
    nEdges = list()



    # 按斜率正负，将线段分为两类
    for i in range(lines.shape[0]):
        for x1, y1, x2, y2 in lines[i]:
            if x2 == x1: # 垂直线
                nEdges.append((x1, y1, x2, y2))
            else:
                k = (y2 - y1) / (x2 - x1)
                # print('k: {}, ({}, {}, {}, {})'.format(k, x1, y1, x2, y2))
                if abs(k) <= 0.1:  # todo: 这里判断不充分
                    pEdges.append((x1, y1, x2, y2))
                else:
                    nEdges.append((x1, y1, x2, y2))

    topEdge = pEdges[0]
    bottomEdge = pEdges[0]
    leftEdge = nEdges[0]
    rightEdge = nEdges[0]

    # 若斜率为正，则找出y最小和y最大的两条线段即可
    for edge in pEdges:
        if edge[1] < topEdge[1]:
            topEdge = edge
        elif edge[1] > bottomEdge[1]:
            bottomEdge = edge
        if edge[3] < topEdge[3]:
            topEdge = edge
        elif edge[3] > bottomEdge[3]:
            bottomEdge = edge

    # 若斜率为负，则找出x最小和x最大的两条线段即可
    for edge in nEdges:
        if edge[0] < leftEdge[0]:
            leftEdge = edge
        elif edge[0] > rightEdge[0]:
            rightEdge = edge
        if edge[2] < leftEdge[2]:
            leftEdge = edge
        elif edge[2] > rightEdge[2]:
            rightEdge = edge
    # print('topEdge: {}, bottomEdge: {}'.format(topEdge, bottomEdge)) 
    # print('leftEdge: {}, rightEdge: {}'.format(leftEdge, rightEdge))
    
    return topEdge, leftEdge, bottomEdge, rightEdge