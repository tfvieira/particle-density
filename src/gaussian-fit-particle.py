#%%
from operator import sub
from utils import *
from imtools import *
from filetools import *
from fitutils import *

#%% Parse arguments
# parser = argparse.ArgumentParser(description='Image file index.')
# parser.add_argument('index', metavar='I', type=int, help='Image index.')
# args = parser.parse_args()
# ind = args.index

#%%
EXPERIMENT_INDEX = 0

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
EXPERIMENT = EXPERIMENTS[EXPERIMENT_INDEX]

# Read configuration parameters from JSON file
CONFIG_FILENAME = os.path.join(CONFIG_PATH, EXPERIMENT + '.json')
with open(CONFIG_FILENAME, 'r') as fp:
    config = json.load(fp)

#%% Read Image data
# for ind in range(19):
IMAGE_INDEX = 9

# Read image
name_list = [os.path.join(config['OUTPUT_PATH'], 'crop_centered', 'crop_centered_' + str(IMAGE_INDEX) + '.tif')]
images = read_list_of_images(name_list)

#%%
image = images[0]
image = resize_normalize_img(image)
image = subtract_bkg_mean(image, patch_size=5)

# ROW = int(image.shape[0]/2)
ROW = 10

plt.figure(1)

x = np.linspace(0,32,32)
plt.plot(x, image[ROW, :], 'k.', label='Image')

O = [16, 16, 10, 2, 3]
y = gaussian(O, x)
y = y / tf.norm(y)
print(tf)
plt.plot(x, y, 'k-', label='Initial O')

losses = fit_gaussian_on_vector(y, tf.convert_to_tensor(O, dtype='float64'))
y_hat = gaussian(O)
plt.plot(x, y_hat, 'b-', label='y_hat')

plt.legend()
plt.title(f'Row: {ROW}.')
plt.show()

plt.figure(2)
plt.plot(losses)
plt.show()

# # O, losses = fit_donut_on_image(img)

# #%%

# # lossFnc = tf.keras.losses.mean_squared_error


# # lst = np.linspace(0.650, 0.675, 100)

# # losses = []

# # y = img

# # for O in lst:

# #     y_hat = donut([O])

# #     loss = tf.reduce_mean(lossFnc(y, y_hat))
# #     losses.append(loss)

# # plt.plot(lst, losses, 'r.')




# #%%
# # #%%

# # # p1, S1, p2, S2 = [
# # #     3.144883514171004,
# # #     13.952410075589007,
# # #     -0.21574063505005947,
# # #     2.2801488724688346
# # # ]#1.8

# # # p1, S1, p2, S2 = [
# # #     3.2,
# # #     13.952410075589007,
# # #     -0.21574063505005947,
# # #     2.2801488724688346
# # # ]#1.87

# # p1, S1, p2, S2 = [
# #     3.0,
# #     13.952410075589007,
# #     -0.21574063505005947,
# #     2.2801488724688346
# # ]# 1.62


# # # p1, S1, p2, S2 = [
# # #     1.0,
# # #     13.952410075589007,
# # #     -0.21574063505005947,
# # #     2.2801488724688346
# # # ]


# # x = np.linspace(-1,1,32)
# # I = p1 * np.exp(-0.5*(((x)**2)*S1)) - p2 * np.exp(-0.5*(((x)**2)*S2 ))
# # y = img[15,:]

# # loss = tf.keras.losses.mean_squared_error(I, y)
# # print(f'{loss.numpy():.2e}')

# # plt.plot(x, I)
# # plt.stem(x, y)
# # plt.show()

# # #%%
# # # O = [
# # #     1.0,
# # #     13.952410075589007,
# # #     -0.21574063505005947,
# # #     2.2801488724688346
# # # ]#Loss = 0.028267

# # O = [
# #     3.144883514171004,
# #     13.952410075589007,
# #     -0.21574063505005947,
# #     2.2801488724688346
# # ]#Loss = 0.387716

# # y_hat = donut(O)

# # loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_hat, img))

# # print(f'Loss = {loss:.6f}')



# # #%%
# # for i in O:
# #     print(i.numpy())

# # #%%
# # cols = I.shape[1]
# # scale = cols/img.shape[1]

# # #%%
# # # u1_1, u1_2, S1, p1, u2_1, u2_2, S2, p2 = O
# # p1, S1, p2, S2 = O
# # # u1_1 = scale * ( u1_1 + img.shape[1]/2)
# # # u1_2 = scale * ( u1_2 + img.shape[0]/2)
# # # u2_1 = scale * ( u2_1 + img.shape[1]/2)
# # # u2_2 = scale * ( u2_2 + img.shape[0]/2)
# # S1   = S1*scale/2
# # S2   = S2*scale/2

# # #%% Draw a circle around the particle
# # I = (I - I.min())/(I.max()-I.min())
# # I = np.uint8(255*I)
# # I = gray2bgr(I)

# # # circle1_center   = (int(u1_1), int(u1_2))
# # circle1_center   = (int(I.shape[1]/2), int(I.shape[0]/2))
# # circle1_radius   = int(np.abs(S1))
# # circle1_color    = (255, 0, 0)
# # circle1_linetype = 2
# # I = cv2.circle(I, circle1_center, circle1_radius, circle1_color, circle1_linetype)

# # # circle2_center   = (int(u2_1), int(u2_2))
# # circle2_center   = (int(I.shape[1]/2), int(I.shape[0]/2))
# # circle2_radius   = int(np.abs(S2))
# # circle2_color    = (0, 0, 255)
# # circle2_linetype = 2
# # I = cv2.circle(I, circle2_center, circle2_radius, circle2_color, circle2_linetype)



# # #%% Export results filenames
# # json_filename        = '../results/10-microns particles-60X/donut_fit/crop_centered_gaussian_fit_'        + str(ind) +'.json'
# # losses_filename      = '../results/10-microns particles-60X/donut_fit/crop_centered_gaussian_fit_losses_' + str(ind) +'.csv'
# # fig_filename         = '../results/10-microns particles-60X/donut_fit/crop_centered_gaussian_fit_'        + str(ind) + '.tif'
# # losses_im_filename   = '../results/10-microns particles-60X/donut_fit/crop_centered_gaussian_fit_losses_' + str(ind) + '.tif'

# # fig = plt.figure()
# # plt.imshow(I, cmap='gray')
# # # plt.contour(yHat)
# # plt.grid(False)
# # # plt.axis(False)
# # plt.xlabel(f'Radius[1,2] = {S1.numpy():.02e}, {S2.numpy():.02e}')
# # plt.savefig(fig_filename)
# # # plt.show()

# # print([x.numpy() for x in O])

# # plt.figure()
# # plt.plot(losses)
# # plt.savefig(losses_im_filename)
# # # plt.show()

# # # Write files containing results
# # gauss_2d = [i.numpy().astype(float) for i in O]
# # gauss_2d[0]
# # write_json(gauss_2d, json_filename)

# # losses = pd.DataFrame(losses)
# # losses.to_csv(losses_filename, header=False, index=False)


# # # #%%
# # # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # # # Make data.
# # # X = np.linspace(-1,1, I.shape[0])
# # # Y = np.linspace(-1,1, I.shape[0])
# # # X, Y = np.meshgrid(X, Y)
# # # # R = np.sqrt(X**2 + Y**2)
# # # # Z = np.sin(R)

# # # # tmp1 = ((X-O[0])**2)*O[2]
# # # # tmp2 = ((X-O[0])*(Y-O[1]))*O[3]
# # # # tmp3 = ((Y-O[1])**2)*O[4]
# # # tmp1 = ((X - O[0])**2)*O[2]
# # # tmp2 = 0
# # # tmp3 = ((Y - O[1])**2)*O[2]
# # # Z = np.exp(-0.5*(tmp1+tmp2+tmp3)) - 2.3

# # # # Plot the surface.
# # # surf = ax.contourf(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# # # surf = ax.plot_surface(X, Y, I, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# # # # Customize the z axis.
# # # # ax.set_zlim(-1.01, 1.01)
# # # ax.zaxis.set_major_locator(LinearLocator(10))
# # # # A StrMethodFormatter is used automatically
# # # # ax.zaxis.set_major_formatter('{x:.02f}')

# # # # Add a color bar which maps values to colors.
# # # fig.colorbar(surf, shrink=0.5, aspect=5)

# # # fig_filename   = '../results/10-microns particles-60X/gaussian_fit/crop_centered_gaussian_fit_result_' + str(ind) + '.tif'
# # # plt.savefig(fig_filename)
# # # print(f'\n\nSaved plot:\n{fig_filename}')

# # # plt.show()



# %%
