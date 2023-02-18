import os

path = "C:/Users/Cauli/Desktop/test_range_image/"  # 待读取的文件夹
path_list = os.listdir(path)
# path_list.sort(key=lambda x: int(x[:-4]))  # 对读取的路径进行排序
# print(path_list)
path_add = []
for filename in path_list:
    path_add.append(os.path.join(path, filename))

# for i in range(len(path_add)-2):
i = 0
while len(path_add) > 2:
    path_new_gt = "C:/Users/Cauli/Desktop/db_test/" + str(i) + '/'
    path_new_input = "C:/Users/Cauli/Desktop/db_test/" + str(i) + '/'
    i += 1
    # 设置旧文件名（就是路径+文件名）
    oldname0 = path_add[0]
    #oldname1 = path_add[1]
    oldname2 = path_add[2]

    os.makedirs(path_new_gt)
    # os.makedirs(path_new_input)
    # 设置新文件名
    newname0 = path_new_gt + 'frame10.png'
    #newname1 = path_new_input + 'frame10i11.png'
    newname2 = path_new_gt + 'frame11.png'

    os.rename(oldname0, newname0)
    #os.rename(oldname1, newname1)
    os.rename(oldname2, newname2)

    path_add.pop(2)
    path_add.pop(1)
    path_add.pop(0)

    print(0)

print(0)