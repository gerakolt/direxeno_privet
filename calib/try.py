import numpy as np

rec0=np.recarray(0, dtype=[
    ('id', 'i8'),
    ('init10', 'i8'),
    ])


rec1=np.recarray(2, dtype=[
    ('id', 'i8'),
    ('init10', 'i8'),
    ])

rec1[0]=0, 10
rec1[1]=1, 11

rec2=np.recarray(2, dtype=[
    ('id', 'i8'),
    ('init10', 'i8'),
    ])

rec2[0]=100, 1000
rec2[1]=101, 1101

print(rec1)
print(rec2)
print(np.append(rec0, rec2))
print(np.append(rec1, rec2)[2])
