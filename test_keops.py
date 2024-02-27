# https://stackoverflow.com/questions/59434659/runtime-error-using-python-library-keops-using-cuda-in-ubuntu18-04
# Testing PyKeops installation
import pykeops

# Changing verbose and mode
pykeops.verbose = True
pykeops.build_type = "Debug"

# Clean up the already compiled files
pykeops.clean_pykeops()

# Test Numpy integration
pykeops.test_numpy_bindings()

# Test Torch integration
pykeops.test_torch_bindings()
