# ctypes: 	is a foreign function library for Python. It provides C compatible data types,
#           and allows calling functions in DLLs or shared libraries.
# 			It can be used to wrap these libraries in pure Python.

import os
import ctypes
from scipy import integrate
from scipy import LowLevelCallable
import numpy as np


# os.path.abspath(path):	Return a normalized absolutized version of the pathname path.
# ctypes.CDLL(): load dynamic link libraries (DLL), on Linux CDLL, on Windows WinDLL or OleDLL
lib = ctypes.CDLL(os.path.abspath('p_b_lib_test.so'))

# ctypes.c_double: ctype data type, in C data type: double, in python data type: float
# restype: specifies the return type -> in this case a C double/ python float
lib.f.restype = ctypes.c_double

# argtypes: It is possible to specify the required argument types of functions exported from DLLs by setting
#           the argtypes attribute (first argument of the function is an integer, second argument is a double and
#           third argument is a void)
# WICHTIG: void Funktion darf keinen Rückgabewert haben!
lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

# ctypes.cast(obj, type): This function is similar to the cast operator in C.
# It returns a new instance of type which points to the same memory block as 'obj'.
# 'type' must be a pointer type, and 'obj' must be an object that can be interpreted as a pointer.
# user_data = ctypes.cast(ctypes.pointer(c), ctypes.c_void_p)

func = LowLevelCallable(lib.f)

f_dsnb = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.15, 0.05, 0, 0])
f_ccatmo = np.array([0.1, 0.2, 0.35, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
f_reactor = np.array([5, 2, 1, 0, 0, 0, 0, 0, 0, 0])
f_data = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0])

fraction = np.array([f_dsnb, f_ccatmo, f_reactor, f_data], dtype='float')


# integrate the function
integral = integrate.nquad(func, [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0, 0.1], [1, 1.1], [1, 1.1], [2, 2.1], [2, 2.1],
                                  [3, 3.1], [3, 3.1], [4, 4.1]])
# print integral-value:
print(integral)
