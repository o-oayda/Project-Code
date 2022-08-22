import healpy as hp

### LaTeX Font for plotting ###
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
##########

hp.projview(healpy_map,
            projection_type='mollweide',
            coord=["G"],
            graticule=True,
            graticule_labels=True,
            color="white",
            cbar=False,
            longitude_grid_spacing=30)
