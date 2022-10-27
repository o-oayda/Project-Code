import healpy as hp
import numpy as np
import time
import math
import scipy.constants as sc
import pandas as pd
import scipy.stats as sts
import dynesty
from dynesty import plotting as dyplot
import multiprocess
import matplotlib
from matplotlib import rc
from funcs import *
from scipy.constants import speed_of_light
from plotting import *
import traceback

####### DYNESTY MODEL FITING FUNCTIONS #######

def model(v, az, pol):
    observerVector = sph2cart((pol, az))
    observerVector2 = np.asarray([observerVector])
    alphaDash = angle(pixelVectors, observerVector2)
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
    tau_val = model(v, az, pol)
    return sum(sts.norm.logpdf(m_new, loc=tau_val, scale=a_new))

####### DETERMINING POINTS REQUIRED #########

t0 = time.time()
newDir = input('Enter directory name: ')

#### Key variables
obsSpeed = 0.001
obsPolar = (0.7,4)
sigma_range = [0.3] # intrinsic uncertainty for each time-scale
nside = 16 # defines pixel density upon pixelisation
points_range = np.linspace(10**6,2*10**7,num=20).astype(int) # range of points
trials = range(1,10) # number of trials to run for

#### Defining other variables
restLambda = ang_freq_to_lambda(1)  # define ang_freq to be 1.
observerVector = sph2cart(obsPolar)
observerVector2 = np.asarray([observerVector])
pol_true, az_true = obsPolar
v_true = obsSpeed

#### Now determine evidences over a range of points
for sigma in sigma_range:
    for k in trials:
        iteration = 0
        ### Create vectors to store evidence values
        fitted_evidence = np.zeros(len(points_range))
        fitted_evidence_unc = np.zeros(len(points_range))
        null_evidence = np.zeros(len(points_range))
        cmb_evidence = np.zeros(len(points_range))
        for n in points_range:
            while True:
                try:
                    # sample points from spherical distribution
                    print('Sampling points, n = ' + str(n) + '...')
                    xi, yi, zi = sample_spherical(n)
                    print('Done.')
                    initialPointsVectors = np.asarray([xi, yi, zi]).T

                    # Apply aberration to points
                    print('Rotating points...')
                    rotatedPoints = transformedPoints(
                        obsSpeed,
                        observerVector2,
                        restLambda,
                        initialPointsVectors,
                        DopplerShift=False)
                    print('Done.')
                    # Place timescales on the sky and dilate depending on point
                    # All rest-frame timescales set to unity
                    tau_true = np.ones(len(rotatedPoints))
                    rotatedPointVectors = np.asarray(
                        [rotatedPoints[0:len(rotatedPoints), 0],
                        rotatedPoints[0:len(rotatedPoints), 1],
                        rotatedPoints[0:len(rotatedPoints), 2]]).T  # as above
                    
                    # use timescale dilation equation
                    alphaDash = angle(rotatedPointVectors, observerVector2)
                    tau_dilated = timeDilation(tau_true,alphaDash,obsSpeed)

                    # Add measurement uncertainty to time-scales
                    tau_error = np.random.normal(loc=0, scale=sigma, size=len(tau_true))
                    
                    # actual measured time-scales in observer frame
                    tau_new = tau_dilated + tau_error  
                    tau_uncertainties = 0.1*np.ones(len(tau_new)) # uncertainty assumed to be 10%

                    print('Averaging points...')
                    # Create pixels and average time-scales with pixelAverage function
                    npix = hp.nside2npix(nside)  # number of pixels
                    pixelVectors, m, a, hpx_map = pixelAverage(
                        nside,
                        rotatedPoints,
                        tau_new,
                        tau_uncertainties)
                    print('Done.')

                    # Apply mask (set points to zero inside mask)
                    theta = np.zeros(len(pixelVectors))
                    for i in range(0, len(pixelVectors)):
                        theta[i] = np.arccos(pixelVectors[i, 2]/math.sqrt(pixelVectors[i, 0]
                            ** 2 + pixelVectors[i, 1]**2 + pixelVectors[i, 2]**2))
                    m[(theta >= np.deg2rad(60)) & (theta <= np.deg2rad(120))] = 0
                    a[(theta >= np.deg2rad(60)) & (theta <= np.deg2rad(120))] = 0

                    # Remove masked pixels which have tau = 0 for model fitting
                    # the new pixels with mask applied are stored in m_new and a_new
                    a_new = a[m!=0]
                    m_new = m[m!=0]
                    pixelVectors = pixelVectors[np.where(m!= 0)]
                    tau_true_pixels = np.ones(npix)
                    tau_true_pixels = tau_true_pixels[np.where(m!= 0)]

                    # Fit parameters using dynesty with multiprocessing
                    with multiprocess.Pool(8) as pool:
                        dsampler = dynesty.NestedSampler(
                        lnlike, prior_transform, ndim=3, pool=pool, queue_size=8)
                        dsampler.run_nested()
                        dresults = dsampler.results

                    # Plot and save cornerplot generated by dynesty
                    fig, axes = dyplot.cornerplot(
                        dresults,
                        truths=[v_true,az_true,pol_true],
                        show_titles=True,
                        title_kwargs={'y': 1.04, 'fontsize' : 17},
                        label_kwargs={'fontsize' : 16},
                        labels=['$v$','$\phi$','$\\theta$'],
                        title_fmt='.3f')
                    
                    plt.savefig('Results/' + newDir + '/corner-n' + str(n) + '-nside' + str(nside) + '.pdf')

                    # Store evidence for fitted hypothesis
                    lnZ = dresults.logz[-1]
                    lnZ_err = dresults.logzerr[-1]
                    fitted_evidence[iteration] = lnZ
                    fitted_evidence_unc[iteration] = lnZ_err
                    print("log(Z) = {0:1.4f} ± {1:1.4}".format(lnZ, lnZ_err))

                    # Compute and store evidence for null hypothesis
                    tau_val_null = 1
                    log_Z_0 = sum(sts.norm.logpdf(m_new, loc=tau_val_null, scale=a_new))
                    null_evidence[iteration] = log_Z_0
                    print("log(Z_0) = {0:1.1f}".format(log_Z_0))

                    # Compute and store evidence for CMB hypothesis
                    pol, az, v = np.pi/2 - (48.253*np.pi/180), 264.021 * \
                    np.pi/180, (369.82*1000)/speed_of_light
                    CMBDipoleVector = sph2cart((pol, az))
                    CMBDipoleVector2 = np.asarray([CMBDipoleVector])
                    alphaDash_CMB = angle(pixelVectors, CMBDipoleVector2)
                    tau_val_CMB = tau_true_pixels * \
                        (1-v*np.cos(alphaDash_CMB))/(np.sqrt(1-v**2))
                    log_Z_CMB = sum(sts.norm.logpdf(m_new, loc=tau_val_CMB, scale=a_new))
                    cmb_evidence[iteration] = log_Z_CMB
                    print("log(Z_CMB) = {0:1.1f}".format(log_Z_CMB))

                    ### Call plotting script to execute plots
                    uncertaintyOnSkyPlot(
                        nside,
                        dresults,
                        az_true,
                        pol_true,
                        n,
                        newDir,
                        topRightQuad=True)
                    
                    numberCountHist(hpx_map,n,newDir)
                    numberCountDensity(hpx_map,m,n,newDir)

                    # Show number of points after mask
                    print('Points (after mask): {}'.format(str(sum(hpx_map))))

                    iteration += 1
                    break
                except ValueError:
                    print(traceback.format_exc())
                    pass
            
        # Unify evidences into dataframe after loop completion and save as csv
        evidence_matrix = np.c_[points_range,null_evidence,cmb_evidence,fitted_evidence,fitted_evidence_unc]
        df = pd.DataFrame(evidence_matrix,columns=['Points','ln(Z_0)','ln(Z_CMB)','ln(Z)','ln(Z_err)'])
        df.to_csv('Results/' + str(newDir) + '/evidences-trials' + str(k) + 'sigma' + str(sigma) + '.csv')
        print(df)

        # Print time taken to run
        t1 = time.time()
        total_time = t1-t0
        print('Time taken (secs): {}'.format(str(total_time)))