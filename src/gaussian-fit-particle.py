#%%
from operator import sub
from utils import *
from imtools import *
from filetools import *
from fitutils import *

#% Parse arguments
# parser = argparse.ArgumentParser(description='Image file index.')
# parser.add_argument('index', metavar='I', type=int, help='Image index.')
# args = parser.parse_args()
# ind = args.index

# Define IO parameters
CONFIG_PATH = 'config'

EXPERIMENTS = [
    '10-microns particles-60X',
    'Isolada 3--3',
    'Isolada 3--2',
    'Isolada-2-10 um',
    'Calibration2_Single Cell',
    'Calibration1_Single Cell',
    'Four-mixing particles together',
    'Several 10-micron-particles together',
    'Calibration 10-microns',
    '30 microns-beads-60X-measuring 2',
    'Calibration-1-4 Cells',
    '3 particles_10 um',
]

# for EXPERIMENT in EXPERIMENTS:
EXPERIMENT_INDEX = 0
EXPERIMENT = EXPERIMENTS[EXPERIMENT_INDEX]

# Read configuration parameters from JSON file
CONFIG_FILENAME = os.path.join(CONFIG_PATH, EXPERIMENT + '.json')
with open(CONFIG_FILENAME, 'r') as fp:
    config = json.load(fp)









#%% Read Image data
diameters = []

for IMAGE_INDEX in range(config['N_IMAGES']):
# IMAGE_INDEX = 10

    # Read image
    imfile = os.path.join(config['OUTPUT_PATH'], 'crop_centered', 'crop_centered_' + str(IMAGE_INDEX) + '.tif')
    image  = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)

    #% Resize and pre-process image
    image = preprocess_image(image, new_shape=NEW_SHAPE)

    #% Fit model to image
    O, losses = fit(image, max_epoch=2000)
    O = [x.numpy() for x in O]
    print(O)

    #%
    radius = 1.4 * (2**O[4])*np.sqrt(O[2])
    diameters.append(radius)
    center = (NEW_SHAPE[0]/2 + O[1], NEW_SHAPE[1]/2 + O[0])
    print(f'Radius: {radius:.6e}.')
    print(f'Center: {center}')

    #% Present the image result
    image = draw_circle(image, center, radius, color=(255,0,0), thickness=1)
    plt.imshow(image)
    plt.grid(False)
    plt.show()

    #% Write radius value to CSV file
    csv_filename = os.path.join(config['OUTPUT_PATH'], 'disks', f'disks_{IMAGE_INDEX}.csv')
    df = pd.DataFrame([radius], columns=['radius'])
    df.to_csv(csv_filename, header=None, index=None)

    #% Save image
    im_out_filename = os.path.join(config['OUTPUT_PATH'], 'disks', f'disks_{IMAGE_INDEX}.tif')
    cv2.imwrite(im_out_filename, image)


diameters_filename = os.path.join(config['OUTPUT_PATH'], 'disks', f'diameters.csv')
df = pd.DataFrame(diameters, columns=['diameters'])
df.to_csv(diameters_filename, header=None, index=None)

plt.plot(df, '-ro')

#%% Read diameters and calibrate
diameters_filename = os.path.join(config['OUTPUT_PATH'], 'disks', f'diameters.csv')
df = pd.read_csv(diameters_filename, header=None)
print(df.T)

scale = np.frompyfunc(lambda x, min, max: (x - min) / (max - min), 3, 1)

y_data = df.values
# y_data = scale(y_data, y_data.min(), y_data.max())

N = len(y_data)
x_data = np.linspace(0,N-1,N)

x_fit = np.linspace(0, N-1, 200)

maxfev = 1000
pars, cov = curve_fit(
    f      = exponential,
    xdata  = x_data.flatten(),
    ydata  = y_data.flatten(),
    p0     = [0, 0], 
    bounds = (-np.inf, np.inf), 
    maxfev = maxfev,
    )

y_fit = exponential(x_fit, pars[0], pars[1])

d = {
    'score': {
        'pars' : {'a': pars[0],  'b' : pars[1]},
        'cov'  : {'a': cov[0,0], 'b' : cov[1,1]},
        },
    }

write_json(d, os.path.join(config['OUTPUT_PATH'], 'calibration', 'calibration.json'))

plot_data_and_single_exponential(x_data + 1, y_data, x_fit + 1, y_fit)
plt.savefig(os.path.join(config['OUTPUT_PATH'], 'calibration', 'calibration.pdf'))


#%%
# # # #%% Export results filenames
# # # json_filename        = '../results/10-microns particles-60X/donut_fit/crop_centered_gaussian_fit_'        + str(ind) +'.json'
# # # losses_filename      = '../results/10-microns particles-60X/donut_fit/crop_centered_gaussian_fit_losses_' + str(ind) +'.csv'
# # # fig_filename         = '../results/10-microns particles-60X/donut_fit/crop_centered_gaussian_fit_'        + str(ind) + '.tif'
# # # losses_im_filename   = '../results/10-microns particles-60X/donut_fit/crop_centered_gaussian_fit_losses_' + str(ind) + '.tif'

# # # fig = plt.figure()
# # # plt.imshow(I, cmap='gray')
# # # # plt.contour(yHat)
# # # plt.grid(False)
# # # # plt.axis(False)
# # # plt.xlabel(f'Radius[1,2] = {S1.numpy():.02e}, {S2.numpy():.02e}')
# # # plt.savefig(fig_filename)
# # # # plt.show()

# # # print([x.numpy() for x in O])

# # # plt.figure()
# # # plt.plot(losses)
# # # plt.savefig(losses_im_filename)
# # # # plt.show()

# # # # Write files containing results
# # # gauss_2d = [i.numpy().astype(float) for i in O]
# # # gauss_2d[0]
# # # write_json(gauss_2d, json_filename)

# # # losses = pd.DataFrame(losses)
# # # losses.to_csv(losses_filename, header=False, index=False)


# # # # #%%
# # # # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # # # # Make data.
# # # # X = np.linspace(-1,1, I.shape[0])
# # # # Y = np.linspace(-1,1, I.shape[0])
# # # # X, Y = np.meshgrid(X, Y)
# # # # # R = np.sqrt(X**2 + Y**2)
# # # # # Z = np.sin(R)

# # # # # tmp1 = ((X-O[0])**2)*O[2]
# # # # # tmp2 = ((X-O[0])*(Y-O[1]))*O[3]
# # # # # tmp3 = ((Y-O[1])**2)*O[4]
# # # # tmp1 = ((X - O[0])**2)*O[2]
# # # # tmp2 = 0
# # # # tmp3 = ((Y - O[1])**2)*O[2]
# # # # Z = np.exp(-0.5*(tmp1+tmp2+tmp3)) - 2.3

# # # # # Plot the surface.
# # # # surf = ax.contourf(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# # # # surf = ax.plot_surface(X, Y, I, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# # # # # Customize the z axis.
# # # # # ax.set_zlim(-1.01, 1.01)
# # # # ax.zaxis.set_major_locator(LinearLocator(10))
# # # # # A StrMethodFormatter is used automatically
# # # # # ax.zaxis.set_major_formatter('{x:.02f}')

# # # # # Add a color bar which maps values to colors.
# # # # fig.colorbar(surf, shrink=0.5, aspect=5)

# # # # fig_filename   = '../results/10-microns particles-60X/gaussian_fit/crop_centered_gaussian_fit_result_' + str(ind) + '.tif'
# # # # plt.savefig(fig_filename)
# # # # print(f'\n\nSaved plot:\n{fig_filename}')

# # # # plt.show()
