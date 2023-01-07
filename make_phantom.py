import dxchange
import numpy as np
from phantom import phantom
from apply_filter_2d_exact import apply_filter_2d_exact
from apply_filter_exact import apply_filter_exact

N = 256 #size of the image for inversion
Ntheta = 3*N//2 #number of angular samples
Ns = N #number of radial samples; Ns==N for this test;
[f,ellipse] = phantom(N)
filter_kind = 'shepp-logan' #ramp,shepp-logan,cosine,cosine2,hamming,hann

# generate filtered phantom with using an exact formula
ff = apply_filter_2d_exact(f,filter_kind,ellipse)
print(np.linalg.norm(ff))
dxchange.write_tiff(ff.astype('float32'),'data/ff',overwrite=True)

# generate filtered Radon data for phantom with using an exact formula
h = apply_filter_exact(Ntheta,Ns,filter_kind,ellipse)
print(np.linalg.norm(h))
dxchange.write_tiff(ff.astype('float32'),'data/fR',overwrite=True)