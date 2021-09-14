#%%
from utils import *

font                   = cv2.FONT_HERSHEY_PLAIN
bottomLeftCornerOfText = (5,28)
fontScale              = .2
fontColor              = (255,255,255)
lineType               = 1

images = []
for ind in range(20):
    # ind = 1

    imfilename = '../results/10-microns particles-60X/crop_centered/crop_centered_' + str(ind) +'.tif'
    I = cv2.imread(imfilename, cv2.IMREAD_GRAYSCALE)

    new_shape = (32, 32)
    I = cv2.resize(I, new_shape, interpolation = cv2.INTER_AREA)
    I = I/255.
    bg = (I[0,0] + I[-1,-1] + I[-1,0] + I[0,-1]) / 4.0

    I = I - bg
    I = I / np.sqrt(np.mean(I**2))
    images.append(I)


    # Read JSON file containing the result of the Gaussian fit.
    json_filename = '../results/10-microns particles-60X/gaussian_fit/crop_centered_gaussian_fit_'        + str(ind) +'.json'
    # Get the Gaussian parameters
    u1, u2, sig11, sig12, sig22 = read_json(json_filename)
    O = [u1, u2, sig11, sig12, sig22]

    # # Compute the Gaussian function
    yHat = gaussian2D(O)
    cv2.circle(I, (int(u1+16), int(u2+16)), int(2*sig11), I.max(), 1)

    # Output results
    # Plot the figure
    plt.figure()
    plt.imshow(I, cmap='gray')
    # plt.contour(yHat)
    plt.grid(False)
    plt.axis(False)
    plt.xlabel(f'{sig11:.02e}')
    # Print Gaussian parameters
    print(f'\nind:{ind}, sig11:{sig11:.2e}, sig12:{sig12:.2e}, sig22:{sig22:.2e}\n')

    fig_filename   = '../results/10-microns particles-60X/gaussian_fit/crop_centered_gaussian_fit_result_' + str(ind) + '.tif'
    plt.savefig(fig_filename)

