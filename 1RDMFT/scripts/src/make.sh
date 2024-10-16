rm ../build/*.pyf 

# build .pyf signature file for compilation
python3 -m numpy.f2py RDMFT.f90 -m RDMFT -h ../build/RDMFT.pyf

# actual compilation, generates .so file, i.e. python-lib
python3 -m numpy.f2py -c --opt='-O3' --f90flags='-Wall -fbackslash -fbounds-check -fopenmp' -lgomp ../build/RDMFT.pyf RDMFT.f90  functionals.f90  

mv *.so ../build
