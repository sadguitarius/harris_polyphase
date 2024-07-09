# polyphase allpass filter design

This is a Python implementation of fred harris' polyphase allpass filter design method. It is is a port of the original Matlab/Octave code from harris' Multirate Signal Processing book. There's a cool Streamlit web app that make it relatively easy to use and to experiment with different filter design parameters.

The easiest way to run this is to install [pixi](https://pixi.sh) for package management and then to run the command `pixi run start` in this directory. Alternatively, install the dependencies and run the command `streamlit run harris_polyphase/tony_des_2.py`

To use the coefficients in a polyphase structure, the basic idea is to implement them as pairs cascades of simple first-order allpass filters. For upsampling, offset one path by one sample, and for downsampling, interleave the outputs. It's necessary to remove the zeros in the coefficent lists since essentially the CPU savings in the polyphase structure come from running the filter cascades at a downsampled rate. N.B. I am not a DSP expert but this does work great. I definitely need to add some example code to demonstrate proper usage.

I still need to clean this up and make it more intuitive, but I figured it would be helpful to others to get this out as is. Suggestions are welcome.