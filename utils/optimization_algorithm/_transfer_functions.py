# @File :_transfer_functions.py
# @Time :2023/3/8   10:34
# @Author : zhaozl
# @Describe :

import numpy as np

def sigmoid(val):
    if val < 0:
        return 1 - 1/(1 + np.exp(val))
    else:
        return 1/(1 + np.exp(-val))


def v_func(val):
    return abs(val/(np.sqrt(1 + val*val)))


def u_func(val):
    alpha, beta = 2, 1.5
    return abs(alpha * np.power(abs(val), beta))


def z_func(val):
    return np.sqrt(1-np.power(5,-abs(np.float64(val))))


def zz_func(val):
    return np.sqrt(1-np.power(8,-abs(np.float64(val))))


def get_trans_function(shape):
    if (shape.lower() == 's'):
        return sigmoid

    elif (shape.lower() == 'v'):
        return v_func

    elif (shape.lower() == 'u'):
        return u_func

    elif (shape.lower() == 'z'):
        return z_func

    elif (shape.lower() == 'zz'):
        return zz_func

    else:
        print('\n[Error!] We don\'t currently support {}-shaped transfer functions...\n'.format(shape))
        exit(1)

