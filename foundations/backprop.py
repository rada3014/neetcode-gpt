import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)


        z = np.matmul(x,w) + b

        y_hat = 1/(1+np.exp(-z))

        # dL/dy_hat
        dL_dyhat = y_hat - y_true

        # dy_hat/dz
        dyhat_dz = y_hat * (1 - y_hat)

        # dz/dw = x
        df_dw = dL_dyhat * dyhat_dz * x

        
        df_db = dL_dyhat * dyhat_dz
        
        return np.round(df_dw,5),np.round(df_db,5)
