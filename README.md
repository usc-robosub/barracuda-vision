# barracuda-vision

## Notes for installation:

CUDA must be installed on the host, and make sure the environment variable `CUDA_HOME` points to the location of the CUDA installation (e.g. `/usr/local/cuda`).

In `darknet_ros`, we must remove different compute_XX if it is not supported by the CUDA version that is on the host. We must also add compute_87 for the Jetson Orin Nano.

Unfortunately these packages are built after container start.