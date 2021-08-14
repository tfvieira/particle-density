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



# for ind in range(config['N_IMAGES']):

ind = 19

# Read image
imfilename = '../results/10-microns particles-60X/crop_centered/crop_centered_' + str(ind) +'.tif'

I = cv2.imread(imfilename, cv2.IMREAD_GRAYSCALE)

new_shape = (32, 32)
I = cv2.resize(I, new_shape, interpolation = cv2.INTER_AREA)
I = I/255.

#%
bg = (I[0,0] + I[-1,-1] + I[-1,0] + I[0,-1]) / 4.0

I = I - bg
I = I / np.sqrt(np.mean(I**2))

plt.imshow(I, cmap='gray')
plt.grid(False)

print(I.max())
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

fig = plt.figure()
plt.imshow(I, cmap='gray')
plt.contour(yHat)
plt.grid(False)
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

# I
# cv2.namedWindow('I', cv2.WINDOW_KEEPRATIO)
# cv2.imshow('I', (255*I).astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.linspace(-1,1, I.shape[0])
Y = np.linspace(-1,1, I.shape[0])
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# tmp1 = ((X-O[0])**2)*O[2]
# tmp2 = ((X-O[0])*(Y-O[1]))*O[3]
# tmp3 = ((Y-O[1])**2)*O[4]
tmp1 = ((X - O[0])**2)*O[2]
tmp2 = 0
tmp3 = ((Y - O[1])**2)*O[2]
Z = np.exp(-0.5*(tmp1+tmp2+tmp3)) - 2.3


# Plot the surface.
surf = ax.contourf(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, I, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

fig_filename   = '../results/10-microns particles-60X/gaussian_fit/crop_centered_gaussian_fit_result_' + str(ind) + '.tif'
plt.savefig(fig_filename)


# plt.show()

