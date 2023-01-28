# Get output for sort task
def sort(lst):
    N = len(lst)
    res = []
    for i in range(N):
        count = 0
        for j in range(N):
            if lst[j] < lst[i]:
                count += 1
    res.append(count)
    return res