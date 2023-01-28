# Get output for lis task
def lis(lst):
    N = len(lst)
    res = []
    for i in range(N):
        num = lst[i]
        max_seq = -1
        for j in range(i):
            if lst[j] < num and max_seq < res[j]:
                max_seq = res[j]
        res.append(max_seq + 1)
    return res