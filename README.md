# Music-Note-Detection
This repository contains code and resources for music note detection using solely signal processing techniques without resorting to any Machine Learning and Deep Learning method. Sound is regenerated after analysis using sinusoidal modeling.
# Language
Code is written in python with a bit of C for improving speed.
# How to use
All the codes are in  `run_code` and implementation of models are in `software/models/`. If set-up is complete you only need to run `run_code/GUI_Interface.py`

In order to use these codes you have to install python 3.* and following modules `ipython`, `numpy`, `matplotlib`, `scipy` and `cython`.

In windows terminal try:

`pip install ipython numpy matplotlib scipy cython`

You need to compile some C functions after downloading whole project. Go to `software/models/utilFunctions_C` and type:
`$ python compileModule.py build_ext --inplace`

If you are facing difficulty running C codes, you may need to use the python version of the codes. Notice the occurence of error and change to python alternative.

For listening music you need two extra module,`pyaudio` and `playsound`.Try:

`pip install playsound pyaudio`
