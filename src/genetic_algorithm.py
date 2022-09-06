from operator import sub

from filetools import *
from fitutils import *
from imtools import *
from utils import *

# Define IO parameters
CONFIG_PATH = 'config'

EXPERIMENT = 'genetic_algorithm'

# Read configuration parameters from JSON file
CONFIG_FILENAME = os.path.join(CONFIG_PATH, EXPERIMENT + '.json')
with open(CONFIG_FILENAME, 'r') as fp:
    config = json.load(fp)

diameters = []

for IMAGE_INDEX in range(config['N_IMAGES']):
# IMAGE_INDEX = 10

    # Read image
    imfile = os.path.join(config['OUTPUT_PATH'], 'crop_centered', 'crop_centered_' + str(IMAGE_INDEX) + '.tif')
    image  = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)

    #% Resize and pre-process image
    image = preprocess_image(image, new_shape=NEW_SHAPE)

    #% Fit model to image
    O, losses = fit(image, max_epoch=1000)
    O = [x.numpy() for x in O]
    print(O)

    radius = 1.4 * (2**O[4])*np.sqrt(O[2])
    diameters.append(radius)
    center = (NEW_SHAPE[0]/2 + O[1], NEW_SHAPE[1]/2 + O[0])
    print(f'Radius: {radius:.6e}.')
    print(f'Center: {center}')

    image = draw_circle(image, center, radius, color=(255,0,0), thickness=1)
    plt.imshow(image)
    plt.grid(False)
    plt.show()

    # csv_filename = os.path.join(config['OUTPUT_PATH'], 'disks', f'disks_{IMAGE_INDEX}.csv')
    # df = pd.DataFrame([radius], columns=['radius'])
    # df.to_csv(csv_filename, header=None, index=None)

    # im_out_filename = os.path.join(config['OUTPUT_PATH'], 'disks', f'disks_{IMAGE_INDEX}.tif')
    # cv2.imwrite(im_out_filename, image)


# diameters_filename = os.path.join(config['OUTPUT_PATH'], 'disks', f'diameters.csv')
# df = pd.DataFrame(diameters, columns=['diameters'])
# df.to_csv(diameters_filename, header=None, index=None)

# plt.plot(df, '-ro')

# diameters_filename = os.path.join(config['OUTPUT_PATH'], 'disks', f'diameters.csv')
# df = pd.read_csv(diameters_filename, header=None)
# print(df.T)

# scale = np.frompyfunc(lambda x, min, max: (x - min) / (max - min), 3, 1)

# y_data = df.values
# y_data = scale(y_data, y_data.min(), y_data.max())

# N = len(y_data)
# x_data = np.linspace(0,N-1,N)

# x_fit = np.linspace(0, N-1, 200)

# maxfev = 1000
# pars, cov = curve_fit(
#     f      = exponential,
#     xdata  = x_data.flatten(),
#     ydata  = y_data.flatten(),
#     p0     = [0, 0], 
#     bounds = (-np.inf, np.inf), 
#     maxfev = maxfev,
#     )

# y_fit = exponential(x_fit, pars[0], pars[1])

# d = {
#     'score': {
#         'pars' : {'a': pars[0],  'b' : pars[1]},
#         'cov'  : {'a': cov[0,0], 'b' : cov[1,1]},
#         },
#     }

# write_json(d, os.path.join(config['OUTPUT_PATH'], 'calibration', 'calibration.json'))

# plot_data_and_single_exponential(x_data + 1, y_data, x_fit + 1, y_fit)
# plt.savefig(os.path.join(config['OUTPUT_PATH'], 'calibration', 'calibration.pdf'))

