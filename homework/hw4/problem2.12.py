import math

N = 40000

n = 0

while n < 10:
    N = int(8/0.05**2 * math.log((4*(2*N)**10+4)/(0.05), math.e))
    print('iteration {}: {}'.format(n+1,N))
    n += 1
