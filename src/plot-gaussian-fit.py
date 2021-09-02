#%%
from utils import *

#%%
for ind in range(20):

    json_filename = '../results/10-microns particles-60X/gaussian_fit/crop_centered_gaussian_fit_'        + str(ind) +'.json'

    u1, u2, sig11, sig12, sig22 = read_json(json_filename)
    O = [u1, u2, sig11, sig12, sig22]

    print(f'ind={ind} sig11={sig11:.4e}')
