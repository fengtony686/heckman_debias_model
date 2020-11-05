from scipy.stats import norm
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools.tools import add_constant
import numpy as np
from tqdm import tqdm
import random
import warnings


warnings.filterwarnings("ignore")


def im(param):
    return np.true_divide(norm.pdf(param), norm.cdf(param))


def dot(u, v):
    return sum([u[i] * v[i] for i in range(min(len(u), len(v)))])


def weight_generating(x, y, param=0):
    print('---------- Generating weights ----------')
    if param == 0:
        model = Probit(y, add_constant(x), missing='drop')
        Probit_model = model.fit()
    elif param == 1:
        while True:
            random_index = [random.choice([True, False]) for _ in range(len(y))]
            X = x[random_index]
            Y = y[random_index]
            model = Probit(Y, add_constant(X), missing='drop')
            Probit_model = model.fit()
            if not np.isnan(Probit_model.params[0]):
                break
    IM_list = []
    for count, i in tqdm(enumerate(y), desc='Computing inverse Mills ratios', ncols=100):
        if param == 0:
            IM_list.append(im(Probit_model.fittedvalues[count]))
        elif param == 1:
            tmp = [1]
            tmp.extend(list(x.iloc[count, ]))
            IM = im(dot(tmp, Probit_model.params))
            IM_list.append(IM)
    weight = np.true_divide(IM_list, np.mean(IM_list))
    return weight
