"""
Don't import this as a module!
Run with 'streamlit run tony_des_2.py'

tony_des_2 forms coefficients of two path recursive polyphase filter
filter can be halfband quadrature mirror, polynomials in Z^2
filter can be bandwidth tuned,        Z^-1 => (1-b*z)/(z-b),       b=(1-tan(bw))/(1+tan(bw))
filter can be center frequency tuned, z^-1 => (-1/z)*(1-cz)/(z-c), c=cos(theta_c)
filter can be zero-packed             Z^-1 ->  z^-k,               k even
order of filter is odd (3,5,7.....)
wo is normalized band edge frequency .5>wo>.25

based on paper "Digital Signal Processing Schemes for Efficient Interpolation and Decimation"
by Valenzuela and Constantinides, IEE Proceedings, Dec 1983
Script file written by fred harris of SDSU. Copyright 2002

original Matlab script file written by fred harris of SDSU. Copyright 2002
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from tonycxx import tonycxx  # type: ignore

coef0 = []
coef1 = []

with st.sidebar:
    N = st.number_input("Prototype Order (Odd)", 1, value=5, step=2, key="N")
    wo = st.number_input("Normalized Band Edge (0.25-0.5)",
                         0.25,
                         0.5,
                         0.3,
                         0.001,
                         "%.6f",
                         key="wo")
    bw = st.number_input("Bandwidth (0.0-0.5)",
                         0.0,
                         0.5,
                         0.25,
                         0.001,
                         "%.6f",
                         key="bw")
    wc = st.number_input("Center Frequency",
                         0.0,
                         0.5,
                         0.0,
                         0.001,
                         "%.6f",
                         key="wc")
    kk = st.number_input("Order (even)", 2, step=2, key="kk")


def do_it():
    """
    Display polyphase allpass filter design UI.
    """
    st.empty()
    if bw == 0.25 and wc == 0:
        mode = 1
    else:
        mode = 2

    aa = 1
    if wc == 0:
        aa = -aa

    roots0, roots1, coef0, coef1, ff0, ff1 = tonycxx(N, wo, bw, wc, kk, mode)
    fig1 = plt.figure(1, (10, 3))
    ax1 = fig1.add_subplot(1, 1, 1)
    eps = np.finfo(float).eps
    ff0[np.where(ff0 == 0)] = eps
    ff1[np.where(ff1 == 0)] = eps
    ax1.plot(20 * np.log10(np.abs(ff0)))
    ax1.plot(20 * np.log10(np.abs(ff1)))
    ax1.set_ylim([-192, 6])
    plt.grid(True)

    fig2 = plt.figure(2, (3, 3))
    zplane = np.exp(1j * 2 * np.pi * np.arange(0, 1.01, 0.01))
    poles0 = np.roots(roots0)
    poles1 = np.roots(roots0)
    zeros = np.roots(
        np.convolve(roots0, roots1[::-1]) -
        aa * np.convolve(roots0[::-1], roots1))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(np.real(poles0), np.imag(poles0), "x", linewidth=2, markersize=8)
    ax2.plot(np.real(poles1), np.imag(poles1), "x", linewidth=2, markersize=8)
    ax2.plot(np.real(zeros), np.imag(zeros), "o", linewidth=2)
    ax2.plot(np.real(zplane), np.imag(zplane), "r")
    ax2.plot([-1.1, 1.1], [0, 0], "k")
    ax2.plot([0, 0], [-1.1, 1.1], "k")
    plt.grid(True)

    st.pyplot(fig1)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(coef0)
    with col2:
        st.write(coef1)
    with col3:
        st.pyplot(fig2)


do_it()
