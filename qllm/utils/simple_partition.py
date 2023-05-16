import math
def partition_a_into_b_bins(a, b):
    remainders = a % b
    ideal_allocation = a // b
    allocation = []
    for i in range(b):
        allocation.append(ideal_allocation) 
    for i in range(remainders):
        allocation[i] += 1 
    return allocation

def partition_a_with_max_b(a, b):
    if a % b == 0:
        return [b] * (a // b)
    else:
        bins = math.ceil(a / b)
        return partition_a_into_b_bins(a, bins)