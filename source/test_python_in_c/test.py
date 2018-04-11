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
#           the argtypes attribute.
# WICHTIG: void Funktion darf keinen Rückgabewert haben!
lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
# lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_double)

# ctypes.c_double: Represents the C double data type
c = ctypes.c_double(10)
d = ctypes.c_double(3)
e = ctypes.c_double(4)


# ctypes.cast(obj, type): This function is similar to the cast operator in C.
# It returns a new instance of type which points to the same memory block as 'obj'.
# 'type' must be a pointer type, and 'obj' must be an object that can be interpreted as a pointer.

# user_data = [ctypes.cast(ctypes.pointer(c), ctypes.c_void_p), ctypes.cast(ctypes.pointer(d), ctypes.c_void_p),
#              ctypes.cast(ctypes.pointer(e), ctypes.c_void_p)]
user_data = ctypes.cast(ctypes.pointer([c, d, e]), ctypes.c_void_p)

func = LowLevelCallable(lib.f, user_data)

# integrate the function
integral = integrate.nquad(func, [[0, 1], [0, 1], [0, 1]])
# print integral-value:
print(integral)
