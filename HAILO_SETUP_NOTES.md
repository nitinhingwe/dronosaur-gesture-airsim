# Dronosaur Hailo Setup Notes

Use this environment for Hailo + AirSim work:

```bash
cd ~/dronosaur_ai/airsim_bridge
source .venv_hailo/bin/activate

This venv was created using:

/usr/bin/python3 -m venv --system-site-packages .venv_hailo

Confirmed working:

python -c "import hailo_platform; print('HAILO OK')"
python -c "import cv2; print('cv2 OK:', cv2.__version__)"

Do not use:

.venv for Hailo work
hailo_workspace/hailo-rpi5-examples/venv_hailo_rpi_examples
pyenv Python for Hailo

Reason:
Hailo Python packages are installed in system Python:
/usr/lib/python3/dist-packages
