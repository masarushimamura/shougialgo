# coding:UTF-8
list1 = [1, 2, 3, 4, 5]
list2 = list1


def func(l2):
    global list1
    l2.append(6)  # list2が変更される
    list1 = [1, 2, 3, 4, 5]
    return l2
print("func=", func(list2))
print("list1=", list1)
print("list2=", list2)


from shougi_algorithm import selectway
print(selectway({0: [1,  0],  2: [0,  1],  4: [2,  0],  3: [0,  2],  1: [1,  2],  6: [2,  2]}, [0,  2,  4], [3,  1,  6]))

