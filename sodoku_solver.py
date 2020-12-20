import numpy as np

'''
    1 求出每个数字为0的位置可以填的数，并将其位置和能填的数分别以key和value的方式
      存储到字典里面
    2 将字典里的数据按照所能填写的数据的多少进行排序，先在能填的数少的里面选取一个
      进行填写
    3 将填写的过程记录到列表里面，这个地方暂时想到的就是用列表记录填写过程
    4 更新1、2步，若出现某一步可填写的数据为空，说明之前某一步的选择有问题，回溯回
      去，更换数值，然后回到步骤1
    5 当所有的数都填完后，退出循环
'''
 
 
def nine(data):
    nine_block = np.zeros([3,3,3,3], dtype = int)
    for i in range(3):
        for j in range(3):
            nine_block[i,j] = data[3*i:3*(i+1),3*j:3*(j+1)]
    return nine_block
 
def num_set(data, nine_block):
    pick_set = {}
    for i in range(9):
        for j in range(9):
            if data[i,j] == 0:
                pick_set[str(i)+str(j)] = set(np.array(range(10))) - \
                (set(data[i,:]) | set(data[:,j]) | \
                set(nine_block[i//3,j//3].ravel()))
    return pick_set
 
def solve(data):    
 
    insert_step = []
    while True:
        
        pick_set = num_set(data, nine(data))
        if len(pick_set) == 0: break
        pick_sort = sorted(pick_set.items(), key = lambda x:len(x[1]))
        item_min = pick_sort[0]
        key = item_min[0]
        value = list(item_min[1])
        insert_step.append((key, value))
        if len(value) != 0:
            data[int(key[0]), int(key[1])] = value[0]
        else:
            insert_step.pop()
            for i in range(len(insert_step)):
                huishuo = insert_step.pop()
                key = huishuo[0]
                insert_num = huishuo[1]
                if len(insert_num) == 1:
                    data[int(key[0]), int(key[1])] = 0
                else:
                    data[int(key[0]), int(key[1])] = insert_num[1]
                    insert_step.append((key, insert_num[1:]))
                    break
    return data    
   
if __name__ == '__main__':
    data =  "0 0 0 0 0 0 0 6 0 \
             0 0 0 0 0 4 7 0 5 \
             5 0 0 0 0 0 1 0 4 \
             1 0 0 0 0 2 4 0 0 \
             0 0 8 0 7 0 0 0 0 \
             0 3 0 6 0 0 0 0 0 \
             2 0 0 0 0 9 0 0 1 \
             0 0 6 0 8 0 0 0 0 \
             0 7 0 3 0 0 0 0 0 "
    data = np.array(data.split(), dtype = int).reshape((9, 9))
    print(data)
    solve(data)