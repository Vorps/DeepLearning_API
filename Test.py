import itertools

"""
    1D 
        A+B
    2D : 
        AA+AB+BA+BB
        
        CA+CB
        AC+BC
    3D : 
        AAA+AAB+ABA+ABB+BAA+BAB+BBA+BBB
        
        CAA+CAB+CBA+CBB
        ACA+ACB+BCA+BCB
        AAC+ABC+BAC+BBC
        
        01
        CCA+CCB
        12
        ACC+BCC
        02
        CAC+CBC
"""
A = "A"
B = "B"
C = "C"

dim = 3

for i in range(dim):
    slices_badge = list(itertools.product(*[[A, B] for _ in range(dim-i)]))
    for indexs in itertools.combinations([0,1,2], i):
        result = []
        for slices in slices_badge:
            slices = list(slices)
            for index in indexs:
                slices.insert(index, C)    
            result.append(tuple(slices))
        print(result)
    print("")
