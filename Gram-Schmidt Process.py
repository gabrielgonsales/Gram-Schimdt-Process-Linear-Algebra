# Gram-Schmidt Process
import numpy as np
import numpy.linalg as la

aprox_zero = 1e-14


def gram_schimdt_basis_1(u_1):
    u_2 = np.array(u_1, dtype=np.float_)  # copy of u_1

    ############# 1st column #############
    u_2[:, 0] = u_2[:, 0] / la.norm(u_2[:, 0])  # normalizing u_2
    u_2[:, 1] = (
        u_2[:, 1] - u_2[:, 1] @ u_2[:, 0] * u_2[:, 0]
    )  # suu_2tract any overlap with our new zeroth vector

    # Now normalise if possiu_2le
    if la.norm(u_2[:, 1]) > aprox_zero:
        u_2[:, 1] = u_2[:, 1] / la.norm(u_2[:, 1])
    else:
        u_2[:, 1] = np.zeros_like(u_2[:, 1])

    ############# 2nd column #############
    u_2[:, 2] = (
        u_2[:, 2] - u_2[:, 2] @ u_2[:, 0] * u_2[:, 0]
    )  # suu_2tract any overlap with the zeroth vector
    u_2[:, 2] = (
        u_2[:, 2] - u_2[:, 2] @ u_2[:, 1] * u_2[:, 1]
    )  # suu_2tract the overlap with the first

    # Now normalise if possiu_2le
    if la.norm(u_2[:, 2]) > aprox_zero:
        u_2[:, 2] = u_2[:, 2] / la.norm(u_2[:, 2])
    else:
        u_2[:, 2] = np.zeros_like(u_2[:, 2])

    ############# 3rd column #############
    u_2[:, 3] = u_2[:, 3] - u_2[:, 3] @ u_2[:, 0] * u_2[:, 0]
    u_2[:, 3] = u_2[:, 3] - u_2[:, 3] @ u_2[:, 1] * u_2[:, 1]
    u_2[:, 3] = u_2[:, 3] - u_2[:, 3] @ u_2[:, 2] * u_2[:, 2]

    # Now normalise if possiu_2le
    if la.norm(u_2[:, 3]) > aprox_zero:
        u_2[:, 3] = u_2[:, 3] / la.norm(u_2[:, 3])
    else:
        u_2[:, 3] = np.zeros_like(u_2[:, 3])

    return u_2


def gram_schimdt_basis_2(u_1):
    u_2 = np.array(u_1, dtype=np.float_)

    for i in range(u_2.shape[1]):
        for j in range(i):
            u_2[:, i] = u_2[:, i] - u_2[:, i] @ u_2[:, j] * u_2[:, j]
        if la.norm(u_2[:, i]) > aprox_zero:
            u_2[:, i] = u_2[:, i] / la.norm(u_2[:, i])
        else:
            u_2[:, i] = np.zeros_like(u_2[:, i])

    return u_2


def dimensions(A):
    dim = np.sum(la.norm(gram_schimdt_basis_2(A), axis=0))
    return dim


# Testing GramSchmidt Functions :
# Test 1
test_array_1 = np.array(
    [[1, 0, 2, 6], [0, 1, 8, 2], [2, 8, 3, 1], [1, -6, 2, 3]], dtype=np.float_
)

print(gram_schimdt_basis_1(test_array_1))

print(gram_schimdt_basis_2(test_array_1))

# Test 2
test_array_2 = np.array([[3, 2, 3], [2, 5, -1], [2, 4, 8], [12, 2, 1]], dtype=np.float_)
print(gram_schimdt_basis_2(test_array_2))

print(dimensions(test_array_2))  # non-square array

# Test 3
test_array_3 = np.array(
    [[6, 2, 1, 7, 5], [2, 8, 5, -4, 1], [1, -6, 3, 2, 8]], dtype=np.float_
)
print(gram_schimdt_basis_2(test_array_3))

print(dimensions(test_array_3))

# Test 4: one vector that is a linear combination of the others.
test_array_4 = np.array([[1, 0, 2], [0, 1, -3], [1, 0, 2]], dtype=np.float_)
print(gram_schimdt_basis_2(test_array_4))

print(dimensions(test_array_4))
