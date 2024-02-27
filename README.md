# Tests file

- `test_cuda.py`
- `test_keops.py`


## Experimented problems

### 1

#### Error: 

> [KeOps] Warning : cuda was detected, but driver API could not be initialized. Switching to cpu only.

#### Solution:

I re-installed CUDA manually but did not add and (properly) link to CUDNN.

Follow [this gist](https://gist.github.com/X-TRON404/e9cab789041ef03bcba13da1d5176e28)

# Timing files

- `timing.py` comparing `pycox` on CPU vs GPU for small-sized problems.