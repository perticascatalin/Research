# Get output for ce task
def ce(lst):
    N = len(lst)
    res = []
    for i in range(N):
        ce_diff = MAXINT
        ce_val = MAXINT
        ce_ind = -1
        for j in range(N):
            if i == j:
                continue
            diff = abs(lst[i] - lst[j])
            if diff < ce_diff or (diff == ce_diff and lst[j] < ce_val):
                ce_diff = diff
                ce_val = lst[j]
                ce_ind = j
        res.append(ce_ind)
    return res