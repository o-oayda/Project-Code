import healpy as hp
import numpy as np
import math
import numba as nb
from scipy.linalg import expm, norm
from scipy.constants import speed_of_light

def sample_spherical(npoints, ndim=3):
    '''
    Draw three variables (x,y,z) from a Gaussian distribution and normalise to unit length.
    '''
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def doppler_shift(angFreq,obsAngle,obsSpeed):
    '''
    For a source with some rest frame angular frequency, determine the Doppler shifted angular frequency a moving observer would perceive given their speed and angle to the observer (as perceived in their frame).
    '''
    # obsAngFreq = angFreq*((np.sqrt(1-obsSpeed**2)/(1 - obsSpeed*np.cos(obsAngle)))) #old function—used wrong equation
    obsAngFreq = angFreq*((obsSpeed*np.cos(obsAngle)+1)/np.sqrt(1-obsSpeed**2))
    return obsAngFreq

def ang_freq_to_lambda(omega):
    '''
    Convert angular frequency to wavelength.
    '''
    wavelength = speed_of_light/(2*math.pi*omega)
    return wavelength

def lambda_to_ang_freq(wavelength):
    '''
    Convert wavelength to angular frequency.
    '''
    omega = speed_of_light/(2*math.pi*wavelength)
    return omega

def vectors_angle(v1, v2):
    '''
    Deprecated slow determination of the angle between two vectors via their dot product.
    '''
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
    '''
    Determine alpha', that is the angle to the source as perceived in the rest frame of the moving observer.
    '''
    abberationAngle = np.arccos((obsSpeed + np.cos(angle))/(obsSpeed*np.cos(angle)+1)) # changed from math.acos to np trig
    return abberationAngle

def unit_norm_vector(v1,v2):
    '''
    Find the unit vector normal to two vectors v1 and v2.
    '''
    normVector = np.cross(v1,v2)
    unitNormVector = normVector/np.linalg.norm(normVector)
    return unitNormVector

def rotateMatrix(axis,angle):
    '''
    Deprecated slow method of determining a rotation matrix corresponding to a rotation by some angle about some axis.
    '''
    rotatedMatrix = expm(np.cross(np.eye(3), axis/norm(axis)*angle))
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

def fastRotateMatrixNew(axis, theta):
    # adapted from top post in
    # https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    Vectorised prescription to return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis, axis=1)[:, np.newaxis]  # vectorised yes
    a = np.cos(theta / 2.0)
    vals = -axis * np.sin(theta / 2.0)[:, np.newaxis]
    b = vals[:, 0]
    c = vals[:, 1]
    d = vals[:, 2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def wavelength_to_rgb(wavelength, gamma=0.8):
    '''
    This converts a given wavelength of light to an 
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
    '''
    Given a point with coordinates x, y and z, as well as a vector corresponding to the observer's motion, determine the angle alpha between the point and the velocity vector in the source rest frame.
    '''
    angles = []
    for i in range(0, len(xi)):
        angles.append(vectors_angle(obsVector, (xi[i], yi[i], zi[i])))
    return angles

#You can disable parallelization with parallel=False
@nb.njit(fastmath=True,error_model="numpy",parallel=False)
def angle(v1,v2):
    '''
    Fast implementation of determining the angle between two vectors through numba.
    '''
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

# Deprecated rotation via quaternions.
# def quaternionRotate(xi,yi,zi,rotAxes,deltaAngles):
#     vector = [xi,yi,zi] # might be an issue
#     vector = np.array([0.] + vector)
#     rot_axis = np.array([0.] + rotAxes)
#     axis_angle = (deltaAngles*0.5) * rot_axis/np.linalg.norm(rot_axis)
#     vec = quat.quaternion(*vector)
#     qlog = quat.quaternion(*axis_angle)
#     q = np.exp(qlog)
#     v_prime = q * vec * np.conjugate(q)
#     v_prime_vec = v_prime.imag
#     return v_prime_vec

def updatePoints(angles, obsSpeed, obsVector, xi, yi, zi):
    '''
    For each point (xi,yi,zi), apply a rotation corresponding to a relativistic abberation due to an observer's movement with respect to the source. Points will congregate along the line of motion and spread out opposite the direction of motion.
    '''
    rotatedPoints = np.zeros((len(xi),3))
    deltaAngles = -(abberation_angle(angles, obsSpeed) - angles)
    rotAxes = unit_norm_vector(np.asarray((xi,yi,zi)).T,obsVector) # turn points xi, yi, zi to array
    rotationMatrices = fastRotateMatrixNew(rotAxes,deltaAngles)
    rotatedPoints = np.einsum('ijk,kj->ki',rotationMatrices,np.asarray([xi,yi,zi]).T)
    
    # deprecated quaternion rotation
    # for i in range(0,len(rotatedPoints)):
    #     rotatedPoints[i] = quaternionRotate(xi[i], yi[i], zi[i], rotAxes[i], deltaAngles[i]) #reshape(len(xi), 1)
    # return rotatedPoints
    
    # old element by element rotation
    # for i in range(0, len(rotatedPoints)):
    #     rotationMatrices = fastRotateMatrix(rotAxes[i],deltaAngles[i])
    #     rotatedPoints[i] = dot(rotationMatrices,(xi[i], yi[i], zi[i]))
    
    rotatedPoints = np.asarray(rotatedPoints, dtype=np.float32)
    return rotatedPoints

def transformedPoints(obsSpeed,obsVector,restLambda,initialPointsVectors,DopplerShift=False):
    '''
    For freshly-generated points, determine the wavelength and new location an observer would perceive due to relativistic Doppler shift/abberation. Setting DopplerShift to false bypasses the observed Doppler shift calculation.
    '''
    angles = angle(initialPointsVectors,obsVector) # fine
    # angles = determineAngleToObs(obsVector,xi,yi,zi)
    if DopplerShift:
        obsLambda = restLambdaToObsLambda(restLambda,angles,obsSpeed)*10**9 # possibly slow

        # no longer need RGB functionality
        # print('Computing RGBs...')
        # RGBs = wavelengthToRGB(obsLambda)
        # print('Done.')

        shift = ((obsLambda*10**(-9) - restLambda)/restLambda)
        # changed from xi, yi, zi to initial point vectors
        rotatedPoints = updatePoints(
            angles, obsSpeed, obsVector, initialPointsVectors[:, 0],
            initialPointsVectors[:, 1], initialPointsVectors[:, 2])
        
        return rotatedPoints, shift, obsLambda
    else:
        rotatedPoints = updatePoints(
            angles, obsSpeed, obsVector, initialPointsVectors[:, 0],
            initialPointsVectors[:, 1], initialPointsVectors[:, 2])

        return rotatedPoints

def cart2sph(obsVector):
    '''
    Transform Cartesian coordinates to spherical coordinates given r = 1 for theta between 0 and pi, and phi between -pi and pi. Return: (theta, phi).
    '''
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

def sph2cart(obsPolar):
    '''
    Transform spherical coordinates in the form (theta, phi) to Cartesian coordinates given r = 1.
    '''
    x = np.sin(obsPolar[0]) * np.cos(obsPolar[1])
    y = np.sin(obsPolar[0]) * np.sin(obsPolar[1])
    z = np.cos(obsPolar[0])
    return (x, y, z)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def pixelAverage(nside,rotatedPoints,tau_new,tau_uncertainties):
    '''
    For each point within a pixel carrying a certain value, average over the pixels and return the average + the uncertainty corresponding to the uncertainties of each point added in quadrature.
    '''
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
    a = np.zeros(hp.nside2npix(nside)) # zero vector of length number o`f pixels on the sky
    m[pixel_indices] = tau_new # match the pixel number of point x_i, y_i, z_i to the corresponding dilated time-scale
    a[pixel_indices] = tau_uncertainties # uncertainties of each pixel

    # create vectors corresponding to location of pixel on the sky
    x_pix, y_pix, z_pix = hp.pix2vec(nside,ordered_pixels)
    pixelVectors = np.asarray([x_pix,y_pix,z_pix]).T # turn into nested array

    return pixelVectors, m, a, hpx_map

# from this guy https://stackoverflow.com/questions/56715139/latex-table-with-uncertainty-from-pandas-dataframe
# from testing, this works as it should for errors
# i.e. with err_points = 1, errors are rounded to 1.s.f.
# which then sets the rounding for the actual mean value
def conv2siunitx(val, err, err_points=1):
    '''
    from this guy https://stackoverflow.com/questions/56715139/latex-table-with-uncertainty-from-pandas-dataframe
    from testing, this works as it should for errors i.e. with err_points = 1, errors are rounded to 1.s.f. which then sets the rounding for the actual mean value
    '''
    val = f'{val:.20e}'.split('e')
    err = f'{err:.20e}'.split('e')
    first_uncertain = int(val[1]) - int(err[1]) + err_points

    my_val = f'{np.round(float(val[0]), first_uncertain-1):.10f}'
    my_err = f'{np.round(float(err[0]), err_points-1):.10f}'.replace('.', '')
    # Avoid 1. and write 1 instead
    if first_uncertain > 1:
        first_uncertain = first_uncertain + 1
    # my addition to capture uncertainties greater than the value itself (hopefully)
    elif first_uncertain == 0:
        first_uncertain = 1

    #handle negative values which take up a position in the string
    if my_val[:1] == '-':
        return (f'{my_val[:first_uncertain+1]}({my_err[:err_points]})e{val[1]}')
    else:
        return(f'{my_val[:first_uncertain]}({my_err[:err_points]})e{val[1]}')


def conv2PlusMinus(val, err):
    '''
    Convert value and associated uncertainty to string with val \pm uncertainty.
    '''
    return str(val) + ' \pm ' + str(err)

def timeDilation(rest_tau,alpha_dash,observer_speed):
    '''
    Computes time dilation of time scales according to the derived result.
    '''
    return rest_tau * (1-observer_speed*np.cos(alpha_dash)) / (np.sqrt(1-observer_speed**2))

def conv2GalacticCoords(az,pol,polar_conv=True):
    '''
    - Converts my coordinate system into the galactic coordinate system.
    - The combination of healpy.projview with a mollweide projection and plt.scatter means negative values of phi move to the left (increasing from 0 degrees) and positive values of phi move to the right (decreasing from 360 degrees) corresponding to a range in phi between -pi and pi.
    - For theta, positive values move to the north pole and negative values move to the source pole (range between -pi/2 and +pi/2).
    - To compensate, this function aligns an azimuthal angle between 0 and 2 pi to its corresponding angle Galactic coordinate l such that both are between 0 and 2 pi or 360 degrees e.g. +pi/2 corresponds to 90 degrees (left on the healpy map).
    - The polar angle also is changed to be between 0 and pi instead of 0 and 2 pi.
    '''
    for i in range(0, len(az)):
        if az[i] > np.pi:
            diff = az[i] - np.pi
            az[i] = -(-np.pi + diff)  # -ve to match moving left as increasing phi
        elif az[i] < -np.pi:
            diff = -np.pi - az[i]
            az[i] = -(np.pi - diff)  # as above
        else:
            az[i] = -az[i]
    
    if not polar_conv:
        return az
    
    elif polar_conv:
        for i in range(0,len(pol)):
            pol[i] = np.pi/2 - pol[i]
        return az, pol