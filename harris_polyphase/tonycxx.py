import numpy as np
from scipy.signal import freqz  # type: ignore


def tonycxx(n_ord, wo, bw, fc, k_order, mode):
    """
    Calculates coefficients for a polyphase halfband allpass filter pair.

    mode = 1 for lowpass prototype

    mode = 2 to bandwidth and center frequency of prototype

    k_order for filter order

    function called by tony_des_2,
    computes paramters of two-path recursive all-pass filter

    also computes parameters when frequency transformed to two-path
    low-pass or band-pass filter

    original Matlab script file written by fred harris of SDSU. Copyright 2002
    """

    wt = 4 * np.pi * (wo-0.25)
    k = np.tan(0.25 * (np.pi - wt))
    k = k * k

    kk = np.sqrt(1 - k*k)

    e = 0.5 * (1 - np.sqrt(kk)) / (1 + np.sqrt(kk))

    q = e + 2 * (e**5) + 15 * (e**9) + 150 * (e**13)
    ww = np.zeros((n_ord-1) // 2)
    aa = ww
    cc = np.r_[ww, 0]
    # step 2

    for i in np.arange((n_ord-1) // 2):
        ww[i] = (2 * (q**0.25) * (np.sin(np.pi * (i+1) / n_ord) -
                                  (q**2) * np.sin(3 * np.pi * (i+1) / n_ord)))
        ww[i] = ww[i] / (1 - 2 * (q * np.cos(2 * np.pi * (i+1) / n_ord) -
                                  (q**4) * np.cos(4 * np.pi * (i+1) / n_ord)))

        wwsq = ww[i] * ww[i]
        aa[i] = np.sqrt(((1 - wwsq*k) * (1 - wwsq/k))) / (1+wwsq)
        cc[i] = (1 - aa[i]) / (1 + aa[i])

    order0 = int(np.floor((n_ord-1) // 4))
    order1 = order0
    if n_ord - 1 - 4*order0 != 0:
        order0 += 1

    coef0 = np.zeros([order0, 3])
    coef1 = np.zeros([order1, 3])

    zz = np.zeros(k_order - 1)

    den0 = np.zeros([order0, len(zz) + 2])
    for i in np.arange(order0):
        den0[i, :] = np.r_[1, zz, cc[2 * i]]
    coef0 = den0

    den1 = np.zeros([order1, len(zz) + 2])
    for i in np.arange(order1):
        den1[i, :] = np.r_[1, zz, cc[2*i + 1]]
    coef1 = den1

    h0 = np.ones(1)
    for i in np.arange(order0):
        h0 = np.convolve(h0, den0[i, :])

    h1 = np.ones(1)
    for i in np.arange(order1):
        h1 = np.convolve(h1, den1[i, :])

    zz2 = np.zeros(k_order // 2)
    h1 = np.r_[h1, zz2]

    g0 = h0[::-1]
    g1 = h1[::-1]

    roots0 = h0
    roots1 = h1

    ww, tp = freqz(g0, h0, 512)
    ww, bt = freqz(g1, h1, 512)

    ff0 = 0.5 * (tp+bt)
    ff1 = 0.5 * (tp-bt)

    if mode == 1:
        return roots0, roots1, coef0, coef1, ff0, ff1
    else:
        tt = np.tan(np.pi * bw)
        b = (1-tt) / (1+tt)

        c = np.cos(2 * np.pi * fc)

        if fc == 0:
            den0 = np.zeros([order0, k_order + 1])
            den1 = np.zeros([order1 + 1, k_order + 1])
            zz = np.zeros((k_order//2) - 1)
            for n in np.arange(order0):
                c00 = 1 + cc[2 * n] * b * b
                c01 = -2 * b * (1 + cc[2 * n])
                c01 = c01 / c00
                c02 = cc[2 * n] + b*b
                c02 = c02 / c00
                den0[n, :] = np.r_[1, zz, c01, zz, c02]
            coef0 = den0

            for n in np.arange(order1):
                c10 = 1 + cc[2*n + 1] * b * b
                c11 = -2 * b * (1 + cc[2*n + 1])
                c11 = c11 / c10
                c12 = cc[2*n + 1] + b*b
                c12 = c12 / c10
                den1[n, :] = np.r_[1, zz, c11, zz, c12]
            zz2 = np.zeros(k_order//2 - 1)
            den1[order1, :1 + k_order//2] = np.r_[1, zz2, -b]
            coef1 = den1

            h0 = np.ones(1)
            for i in np.arange(order0):
                h0 = np.convolve(h0, den0[i, :])

            h1 = np.r_[1, zz2, -b]
            for i in np.arange(order1):
                h1 = np.convolve(h1, den1[i, :])

            g0 = h0[::-1]
            g1 = h1[::-1]
            roots0 = h0
            roots1 = h1

            ww, tp = freqz(g0, h0, 512)
            ww, bt = freqz(g1, h1, 512)

            ff0 = 0.5 * (tp+bt)
            ff1 = 0.5 * (tp-bt)

        else:
            den0 = np.zeros([order0, 2*k_order + 1])
            den1 = np.zeros([order1 + 1, 2*k_order + 1])

            zz = np.zeros((k_order//2) - 1)

            for n in np.arange(order0):
                c00 = 1 + cc[2 * n] * b * b
                c01 = -2 * c * (1+b) * (1 + cc[2 * n] * b)
                c01 = c01 / c00
                c02 = (1 + cc[2 * n]) * (c * c * (1 + b*b) + 2 * b * (1 + c*c))
                c02 = c02 / c00
                c03 = -2 * c * (1+b) * (cc[2 * n] + b)
                c03 = c03 / c00
                c04 = cc[2 * n] + b*b
                c04 = c04 / c00
                den0[n, :] = np.r_[1, zz, c01, zz, c02, zz, c03, zz, c04]
            coef0 = den0

            for n in np.arange(order1):
                c10 = 1 + cc[2*n + 1] * b * b
                c11 = -2 * c * (1+b) * (1 + cc[2*n + 1] * b)
                c11 = c11 / c10
                c12 = (1 + cc[2*n + 1]) * (c * c * (1 + b*b) + 2 * b *
                                           (1 + c*c))
                c12 = c12 / c10
                c13 = -2 * c * (1+b) * (cc[2*n + 1] + b)
                c13 = c13 / c10
                c14 = cc[2*n + 1] + b*b
                c14 = c14 / c10
                den1[n, :] = np.r_[1, zz, c11, zz, c12, zz, c13, zz, c14]
            zz2 = np.zeros(k_order//2 - 1)
            den1[order1, :1 + k_order] = np.r_[1, zz2, -c * (1+b), zz2, b]
            coef1 = den1

            h0 = np.ones(1)
            for i in np.arange(order0):
                h0 = np.convolve(h0, den0[i, :])

            h1 = np.r_[1, zz, -c * (1+b), zz, b]
            for i in np.arange(order1):
                h1 = np.convolve(h1, den1[i, :])

            g0 = h0[::-1]
            g1 = h1[::-1]
            roots0 = h0
            roots1 = h1

            ww, tp = freqz(g0, h0, 512)
            ww, bt = freqz(g1, h1, 512)

            ff0 = 0.5 * (tp-bt)
            ff1 = 0.5 * (tp+bt)

    return roots0, roots1, coef0, coef1, ff0, ff1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    roots0, roots1, coef0, coef1, ff0, ff1 = tonycxx(5, 0.3, 0.25, 0, 2, 2)

    # print(roots0,roots1,coef0,coef1,ff0,ff1)
    print("coef0:")
    print(coef0)
    print("coef1:")
    print(coef1)

    fig, ax = plt.subplots()
    ax.plot(20 * np.log10(np.abs(ff0)))
    plt.show()
