from multiprocessing.sharedctypes import Value
import healpy as hp
import numpy as np
from numpy import cross, eye, dot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import time
import math
import scipy.constants as sc
from scipy.linalg import expm, norm
import pandas as pd
from matplotlib.widgets import Slider, Button
import colorcet as cc
import numba as nb
from astropy.coordinates import SkyCoord
import matplotlib.cm as cm
import scipy.stats as sts
import dynesty
from dynesty import plotting as dyplot
import multiprocess
import matplotlib
from matplotlib import rc
import quaternion as quat

##### DEFINED FUNCTIONS #######

# Three variables drawn from Gaussian distribution
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

# Doppler shift a given angular frequency given angle to observer in source frame and observer's speed 
def doppler_shift(angFreq,obsAngle,obsSpeed):
    # obsAngFreq = angFreq*((np.sqrt(1-obsSpeed**2)/(1 - obsSpeed*np.cos(obsAngle)))) #old function—used wrong equation
    obsAngFreq = angFreq*((obsSpeed*np.cos(obsAngle)+1)/np.sqrt(1-obsSpeed**2))
    return obsAngFreq

def ang_freq_to_lambda(omega):
    wavelength = sc.c/(2*math.pi*omega)
    return wavelength

def lambda_to_ang_freq(wavelength):
    omega = sc.c/(2*math.pi*wavelength)
    return omega

def vectors_angle(v1, v2):
    if (np.dot(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)) <= 1 and (np.dot(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)) >= -1:
        return math.acos((np.dot(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    elif (np.dot(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)) < -1:
        print('Less:' + str((np.dot(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2))))
        return math.acos(-1)
    elif (np.dot(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)) > 1:
        print('Greater:' + str((np.dot(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2))))
        return math.acos(1)

# def vectors_angle(v1,v2):
#     angle = math.acos((np.dot(v1,v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)))
#     return angle

def abberation_angle(angle,obsSpeed):
    abberationAngle = np.arccos((obsSpeed + np.cos(angle))/(obsSpeed*np.cos(angle)+1)) # changed from math.acos to np trig
    return abberationAngle

def unit_norm_vector(v1,v2):
    normVector = np.cross(v1,v2)
    unitNormVector = normVector/np.linalg.norm(normVector)
    return unitNormVector

def rotateMatrix(axis,angle):
    rotatedMatrix = expm(cross(eye(3), axis/norm(axis)*angle))
    return rotatedMatrix

def fastRotateMatrix(axis,theta):
    # from top post in
    # https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def wavelength_to_rgb(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return [int(R), int(G), int(B)]

def determineAngleToObs(obsVector, xi, yi, zi):
    # angles between all vectors and observer vector from dot product
    angles = []
    for i in range(0, len(xi)):
        angles.append(vectors_angle(obsVector, (xi[i], yi[i], zi[i])))
    return angles

# NEW ANGLE FUNCTION
#You can disable parallelization with parallel=False
@nb.njit(fastmath=True,error_model="numpy",parallel=False)
def angle(v1,v2):
    #Check the dimensions, this may also have an effect on SIMD-vectorization
    assert v1.shape[1]==3
    assert v2.shape[1]==3
    res=np.empty(v1.shape[0])

    for i in nb.prange(v1.shape[0]): #v2 is now the observer vector
        dot=0.
        a=0.
        b=0.
        for j in range(3):
            dot+=v1[i,j]*v2[0,j]
            a+=v1[i,j]**2
            b+=v2[0,j]**2
        res[i]=np.arccos(dot/(np.sqrt(a*b)))
        # for j in range(3):
        #     dot+=v1[i,j]*v2[i,j]
        #     a+=v1[i,j]**2
        #     b+=v2[i,j]**2
        # res[i]=np.arccos(dot/(np.sqrt(a*b)))
    return res

def restLambdaToObsLambda(restLambda, angles, obsSpeed):
    # lambda values as perceived by moving observer
    restAngFreq = lambda_to_ang_freq(restLambda) # fine
    obsLambda = (ang_freq_to_lambda(doppler_shift(restAngFreq, angles, obsSpeed))) # probably slow
    return obsLambda

def wavelengthToRGB(obsLambda):
    RGBs = []
    for i in range(0, len(obsLambda)):
        RGBs.append(wavelength_to_rgb(obsLambda[i]))
    RGBs = np.asarray(RGBs, dtype=np.float32)
    return RGBs

def quaternionRotate(xi,yi,zi,rotAxes,deltaAngles):
    vector = [xi,yi,zi] # might be an issue
    
    vector = np.array([0.] + vector)
    rot_axis = np.array([0.] + rotAxes)
    axis_angle = (deltaAngles*0.5) * rot_axis/np.linalg.norm(rot_axis)

    vec = quat.quaternion(*vector)
    qlog = quat.quaternion(*axis_angle)
    q = np.exp(qlog)

    v_prime = q * vec * np.conjugate(q)

    v_prime_vec = v_prime.imag

    return v_prime_vec

def updatePoints(angles, obsSpeed, obsVector, xi, yi, zi):
    ## Now we try to congregate points along the source of motion due to abberation
    # angle difference
    rotatedPoints = np.zeros((len(xi),3))
    deltaAngles = -(abberation_angle(angles, obsSpeed) - angles)
    rotAxes = unit_norm_vector(np.asarray((xi,yi,zi)).T,obsVector) # turn points xi, yi, zi to array
    
    # for i in range(0,len(rotatedPoints)):
    #     rotatedPoints[i] = quaternionRotate(xi[i], yi[i], zi[i], rotAxes[i], deltaAngles[i]) #reshape(len(xi), 1)
    # return rotatedPoints

    for i in range(0, len(rotatedPoints)):
        rotationMatrices = fastRotateMatrix(rotAxes[i],deltaAngles[i])
        rotatedPoints[i] = dot(rotationMatrices,(xi[i], yi[i], zi[i]))
    
    rotatedPoints = np.asarray(rotatedPoints, dtype=np.float32)
    return rotatedPoints

def transformedPoints(obsSpeed,obsVector,restLambda,initialPointsVectors):
    angles = angle(initialPointsVectors,obsVector) # fine
    # angles = determineAngleToObs(obsVector,xi,yi,zi)
    obsLambda = restLambdaToObsLambda(restLambda,angles,obsSpeed)*10**9 # possibly slow
    RGBs = wavelengthToRGB(obsLambda)
    shift = ((obsLambda*10**(-9) - restLambda)/restLambda)
    # changed from xi, yi, zi to initial point vectors
    rotatedPoints = updatePoints(
        angles, obsSpeed, obsVector, initialPointsVectors[:, 0],
        initialPointsVectors[:, 1], initialPointsVectors[:, 2])
    return rotatedPoints, shift, obsLambda

def cart2sph(obsVector):
    hxy = np.hypot(obsVector[0],obsVector[1])
    if obsVector[2] == 0:
        theta = np.pi/2
    else:
        theta = np.arccos(obsVector[2]/math.sqrt(obsVector[0]**2 + obsVector[1]**2 + obsVector[2]**2))
    if obsVector[0] > 0:
        atan = np.arctan(obsVector[1]/obsVector[0])
        phi = atan
    elif obsVector[0] < 0 and obsVector[1] >= 0:
        atan = np.arctan(obsVector[1]/obsVector[0])
        phi = atan + np.pi
    elif obsVector[0] < 0 and obsVector[1] < 0:
        atan = np.arctan(obsVector[1]/obsVector[0])
        phi = atan - np.pi
    elif obsVector[0]==0 and obsVector[1] > 0:
        phi = np.pi/2
    elif obsVector[0]==0 and obsVector[1] < 0:
        phi = -np.pi/2
    elif obsVector[0] == 0 and obsVector[1] == 0:
        phi = 0
    return (theta, phi)

def sph2cart(obsPolar): #r=1
    x = np.sin(obsPolar[0]) * np.cos(obsPolar[1])
    y = np.sin(obsPolar[0]) * np.sin(obsPolar[1])
    z = np.cos(obsPolar[0])
    return (x, y, z)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def pixelAverage(nside,):
    npix = hp.nside2npix(nside) # number of pixels
    ordered_pixels = np.arange(npix) # ordered vector of pixel numbers
    pixel_indices = hp.vec2pix(nside, rotatedPoints[0:len(rotatedPoints), 0], rotatedPoints[0:len(rotatedPoints), 1], rotatedPoints[0:len(
        rotatedPoints), 2])  # Gives me the pixel number corresponding to that particular data vector

    # find repeated pixel indices (more than one point in a pixel), and average over
    idx_sort = np.argsort(pixel_indices) # creates an array of indices, sorted by unique element
    sorted_pixel_indices = pixel_indices[idx_sort] # sorts records array so all unique elements are together 

    vals, idx_start, count = np.unique(
        sorted_pixel_indices, return_counts=True, return_index=True) # returns the unique values, the index of the first occurrence of a value, and the count for each element
    res = np.split(idx_sort, idx_start[1:]) # splits the indices into separate arrays

    #filter them with respect to their size, keeping only items occurring more than once
    vals = vals[count > 1]
    res = filter(lambda x: x.size > 1, res)
    vals # actual values of the repeated index
    repeated_indices = tuple(res) # tuple containing the indices of each repeated pixel index in pixel_indices (length n) 

    # Source density map
    idx, counts = np.unique(pixel_indices, return_counts=True)
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[idx] = counts
    ###

    # returns average value of tau for pixels where the index is repeated, then replaces tau value for repeated pixels with average
    print('Averaging over pixels')
    tau_averages = []
    tau_new_uncertainties = []
    for i in range(0,len(repeated_indices)):
        taus = []
        tau_unc = []
        for j in range(0,len(repeated_indices[i])):
            taus.append(tau_new[repeated_indices[i][j]])
            tau_unc.append(tau_uncertainties[repeated_indices[i][j]])
        tau_averages.append(np.mean(taus))
        tau_new_uncertainties.append(
            1/(len(tau_unc))*np.sqrt(np.sum(np.asarray(tau_unc)**2)))
        tau_new[repeated_indices[i]] = tau_averages[i] # replace corresponding tau_new for repeated pixels with the average tau
        # replace corresponding tau_uncertainties for repeated pixels with the errors added in quadrature
        tau_uncertainties[repeated_indices[i]] = tau_new_uncertainties[i]

    m = np.zeros(hp.nside2npix(nside)) # zero vector of length number of pixels on the sky
    a = np.zeros(hp.nside2npix(nside)) # zero vector of length number of pixels on the sky
    m[pixel_indices] = tau_new # match the pixel number of point x_i, y_i, z_i to the corresponding dilated time-scale
    a[pixel_indices] = tau_uncertainties # uncertainties of each pixel

    # create vectors corresponding to location of pixel on the sky
    x_pix, y_pix, z_pix = hp.pix2vec(nside,ordered_pixels)
    pixelVectors = np.asarray([x_pix,y_pix,z_pix]).T # turn into nested array

    return pixelVectors, m, a, hpx_map

###### Dynesty model fitting functions

def model(v, az, pol):
    observerVector = sph2cart((pol, az))
    observerVector2 = np.asarray([observerVector])
    alphaDash = angle(pixelVectors, observerVector2)
    # alphaDash = determineAngleToObs(observerVector,x_pix,y_pix,z_pix) # OLD function; find what the Doppler shift should be at the pixel location (presumably the centre of the pixel)
    model = tau_true_pixels * \
        (1-v*np.cos(alphaDash))/(np.sqrt(1-v**2))  # assumes knowledge of actual time-scale
    return model

def prior_transform(uTheta):
    uV, uAz, uPol = uTheta
    v = 0.01*uV  # uniform between 0 and 0.01
    az = 2*np.pi*uAz  # uniform between 0 and 2*pi # np.pi + np.pi*uAz
    pol = uPol*np.pi  # uniform between 0 and pi
    return v, az, pol

def lnlike(Theta):
    v, az, pol = Theta
    # turn az and pol into alpha' for this omega
    tau_val = model(v, az, pol)
    # changed tau_new to m, tau_uncertainties to a
    return sum(sts.norm.logpdf(m_new, loc=tau_val, scale=a_new))

#######################################

####### DETERMINING POINTS REQUIRED #########

t0 = time.time()

### LaTeX Font for plotting ###
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
##########

#### Initial variables

restLambda = ang_freq_to_lambda(1)  # define ang_freq to be 1.
obsSpeed = 0.001
obsPolar = (0.7, 4)  # pol, az nominally 0.7,4
observerVector = sph2cart(obsPolar)
observerVector2 = np.asarray([observerVector])
pol_true, az_true = obsPolar
v_true = obsSpeed
sigma = 0.3 # intrinsic uncertainty for each time-scale
nside = 16 # defines pixel density of projection
# range of points to iterate over
# points_range = np.linspace(5*10**5, 10**7, num=11).astype(int)
points_range = np.linspace(5*10**6, 20*10**6, num=16).astype(int)
# (np.linspace(5*10**5, 10**7, num=11).astype(int)) # 500,000 to 10 million in increments of 950,000
iteration = 0

### Create vectors for evidence values
fitted_evidence = np.zeros(len(points_range))
fitted_evidence_unc = np.zeros(len(points_range))
null_evidence = np.zeros(len(points_range))
cmb_evidence = np.zeros(len(points_range))

#### Now determine evidences over a range of points

for n in points_range:
    while True:
        try:
            # sample points from spherical distribution
            print('Sampling points, n = ' + str(n))
            xi, yi, zi = sample_spherical(n)
            initialPointsVectors = np.asarray([xi, yi, zi]).T

            # Apply abberation to points
            print('Rotating points')
            rotatedPoints, RGBs, obsLambda = transformedPoints(
                obsSpeed, observerVector2, restLambda, initialPointsVectors)
            print('Points rotated')
            # Place time-scales on the sky and dilate depending on point
            # rest frame tau values (unity for all points)
            tau_true = np.ones(len(rotatedPoints))
            rotatedPointVectors = np.asarray([rotatedPoints[0:len(
                rotatedPoints), 0], rotatedPoints[0:len(rotatedPoints), 1], rotatedPoints[0:len(rotatedPoints), 2]]).T  # as above
            alphaDash = angle(rotatedPointVectors, observerVector2)
            tau_dilated = tau_true * \
                (1-obsSpeed*np.cos(alphaDash)) / \
                (np.sqrt(1-obsSpeed**2))  # time-dilated taus

            # Add measurement uncertainty to time-scales
            tau_error = np.random.normal(loc=0, scale=sigma, size=len(tau_true))
            tau_new = tau_dilated + tau_error  # actual measured time-scales in observer frame
            observed_data = tau_new #???
            tau_uncertainties = 0.1*np.ones(len(tau_new)) # uncertainty assumed to be 10%

            # Create pixels and average time-scales across pixels with pixelAverage function
            npix = hp.nside2npix(nside)  # number of pixels
            pixelVectors, m, a, hpx_map = pixelAverage(nside)

            # Apply mask
            theta = np.zeros(len(pixelVectors))
            for i in range(0, len(pixelVectors)):
                theta[i] = np.arccos(pixelVectors[i, 2]/math.sqrt(pixelVectors[i, 0]
                    ** 2 + pixelVectors[i, 1]**2 + pixelVectors[i, 2]**2))
            
            # Set to 0 where inside mask
            m[(theta >= np.deg2rad(60)) & (theta <= np.deg2rad(120))] = 0
            a[(theta >= np.deg2rad(60)) & (theta <= np.deg2rad(120))] = 0

            # Remove masked pixels which have tau = 0 for model fitting
            # the new pixels with mask applied are stored in m_new and a_new
            a_new = a[m!=0]
            m_new = m[m!=0]
            pixelVectors = pixelVectors[np.where(m!= 0)]
            tau_true_pixels = np.ones(npix)
            tau_true_pixels = tau_true_pixels[np.where(m!= 0)]

            # Fit parameters using dynesty and multiprocessing
            with multiprocess.Pool(8) as pool:
                dsampler = dynesty.DynamicNestedSampler(
                lnlike, prior_transform, ndim=3, pool=pool, queue_size=8)
                dsampler.run_nested()
                dresults = dsampler.results

            # Plot and save cornerplot generated by dynesty
            fig, axes = dyplot.cornerplot(dresults,
                truths=[v_true,az_true,pol_true],
                show_titles=True,
                title_kwargs={'y': 1.04},
                labels=['$v$','$\phi$','$\\theta$'],
                title_fmt='.3f')
            plt.savefig('Results/' + 'corner-n' + str(n) + '-nside' + str(nside) + '.pdf')

            # Fitted hypothesis
            lnZ = dresults.logz[-1]
            lnZ_err = dresults.logzerr[-1]
            fitted_evidence[iteration] = lnZ
            fitted_evidence_unc[iteration] = lnZ_err
            print("log(Z) = {0:1.4f} ± {1:1.4}".format(lnZ, lnZ_err))

            # Null hypothesis
            tau_val_null = 1
            log_Z_0 = sum(sts.norm.logpdf(m_new, loc=tau_val_null, scale=a_new))
            null_evidence[iteration] = log_Z_0
            print("log(Z_0) = {0:1.1f}".format(log_Z_0))

            # CMB hypothesis
            pol, az, v = np.pi/2 - (48.253*np.pi/180), 264.021 * \
            np.pi/180, (369.82*1000)/sc.c
            CMBDipoleVector = sph2cart((pol, az))
            CMBDipoleVector2 = np.asarray([CMBDipoleVector])
            alphaDash_CMB = angle(pixelVectors, CMBDipoleVector2)
            tau_val_CMB = tau_true_pixels * \
                (1-v*np.cos(alphaDash_CMB))/(np.sqrt(1-v**2))
            log_Z_CMB = sum(sts.norm.logpdf(m_new, loc=tau_val_CMB, scale=a_new))
            cmb_evidence[iteration] = log_Z_CMB
            print("log(Z_CMB) = {0:1.1f}".format(log_Z_CMB))

            ### Call plotting script to execute plots for this specific number of points
            # currently don't want to generate any more plots...
            # import plotting

            # plotting.uncertaintyOnSkyPlot(nside,dresults,az_true,pol_true,n)
            # plotting.numberCountHist(hpx_map,n)
            # plotting.numberCountDensity(hpx_map,m,n)

            iteration += 1
            break
        except ValueError:
                pass
    
# Unify evidences into dataframe after loop completion

evidence_matrix = np.c_[points_range,null_evidence,cmb_evidence,fitted_evidence,fitted_evidence_unc]
df = pd.DataFrame(evidence_matrix,columns=['Points','ln(Z_0)','ln(Z_CMB)','ln(Z)','ln(Z_err)'])
df.to_csv('Results/evidences.csv')

print(df)

t1 = time.time()
total_time = t1-t0
print(total_time)