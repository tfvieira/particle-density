#%%
from utils import *

# #%%
# y = gaussian([-1.2, 0, 2.2**2, 0.8**2, 0.5])

# u1 = tf.Variable(0.0)
# sig21 = tf.Variable(1.0)
# u2 = tf.Variable(0.0)
# sig22 = tf.Variable(1.0)
# p = tf.Variable(0.0)
# O = [u1, u2, sig21, sig22, p]

# losses = fit(gaussian, O, y, max_epoch = 300)
# yHat = gaussian([u1, u2, sig21, sig22, p])

# plt.plot(y,    'b.')
# plt.plot(yHat, 'r-')
# plt.figure()
# plt.plot(losses)



#%
# for ind in range(config['N_IMAGES']):

ind = 15

# Read image
imfilename = '../results/10-microns particles-60X/crop_centered/crop_centered_' + str(ind) +'.tif'

I = cv2.imread(imfilename, cv2.IMREAD_GRAYSCALE)

new_shape = (32, 32)
I = cv2.resize(I, new_shape, interpolation = cv2.INTER_AREA)
I = I/255.

u1    = tf.Variable(0.0)
u2    = tf.Variable(0.0)
sig11 = tf.Variable(1.0)
sig12 = tf.Variable(1.0)
sig22 = tf.Variable(1.0)

O = [u1, u2, sig11, sig12, sig22]

losses = fit(gaussian2D, O, I, max_epoch = 200)

yHat = gaussian2D(O)

# Export results filenames
json_filename   = '../results/10-microns particles-60X/gaussian_fit/crop_centered_gaussian_fit_'        + str(ind) +'.json'
losses_filename = '../results/10-microns particles-60X/gaussian_fit/crop_centered_gaussian_fit_losses_' + str(ind) +'.csv'
fig_filename   = '../results/10-microns particles-60X/gaussian_fit/crop_centered_gaussian_fit_' + str(ind) + '.tif'
losses_im_filename   = '../results/10-microns particles-60X/gaussian_fit/crop_centered_gaussian_fit_losses_' + str(ind) + '.tif'

# Plot and save results
fig = plt.figure()
plt.imshow(I, cmap='gray')
plt.contour(yHat)
plt.show()
plt.savefig(fig_filename)
plt.figure()
plt.plot(losses)
plt.savefig(losses_im_filename)

# Write files containing results
gauss_2d = [i.numpy().astype(float) for i in O]
gauss_2d[0]
write_json(gauss_2d, json_filename)

losses = pd.DataFrame(losses)
losses.to_csv(losses_filename, header=False, index=False)

