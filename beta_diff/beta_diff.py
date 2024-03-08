import numpy as np
from scipy import integrate
from scipy.special import betaln

class beta_diff:
    def __init__(self, alpha1, beta1, alpha2, beta2):
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2

    @staticmethod
    def jnt_dst(s, t, alpha1, beta1, alpha2, beta2, betaln1, betaln2):
        val = (alpha1 - 1) * np.log(s+t) + (beta1 - 1) * np.log(1 - s - t) + (alpha2 - 1) * np.log(s) + (beta2 - 1) * np.log(1 - s) - betaln1 - betaln2
        return np.exp(val)

    def pdf(self, t):
        betaln1 = betaln(self.alpha1, self.beta1)
        betaln2 = betaln(self.alpha2, self.beta2)
        if t <= -1 or t >= 1:
            return 0
        elif t <= 0:
            val = integrate.quad(
                beta_diff.jnt_dst,
                -t,
                1,
                args=(t, self.alpha1, self.beta1, self.alpha2, self.beta2, betaln1, betaln2),
            )[0]
            return val
        else:
            val = integrate.quad(
                beta_diff.jnt_dst,
                0,
                1 - t,
                args=(t, self.alpha1, self.beta1, self.alpha2, self.beta2, betaln1, betaln2),
            )[0]
            return val

    def cdf(self, t):
        betaln1 = betaln(self.alpha1, self.beta1)
        betaln2 = betaln(self.alpha2, self.beta2)
        if t <= -1:
            return 0
        elif t >= 1:
            return 1
        elif t <= 0:
            val = integrate.dblquad(
                beta_diff.jnt_dst,
                -1,
                t,
                lambda x: -x,
                1,
                args=(self.alpha1, self.beta1, self.alpha2, self.beta2, betaln1, betaln2),
            )[0]
            return val
        else:
            val = (
                integrate.dblquad(
                    beta_diff.jnt_dst,
                    -1,
                    0,
                    lambda x: -x,
                    1,
                    args=(self.alpha1, self.beta1, self.alpha2, self.beta2, betaln1, betaln2),
                )[0]
                + integrate.dblquad(
                    beta_diff.jnt_dst,
                    0,
                    t,
                    0,
                    lambda x: 1 - x,
                    args=(self.alpha1, self.beta1, self.alpha2, self.beta2, betaln1, betaln2),
                )[0]
            )
            return val

    @staticmethod
    def jnt_dst_for_mean(s, t, alpha1, beta1, alpha2, beta2, betaln1, betaln2):
        val = (alpha1 - 1) * np.log(s+t) + (beta1 - 1) * np.log(1 - s - t) + (alpha2 - 1) * np.log(s) + (beta2 - 1) * np.log(1 - s) - betaln1 - betaln2
        return t * beta_diff.jnt_dst(s, t, alpha1, beta1, alpha2, beta2, betaln1, betaln2)

    def mean(self):
        betaln1 = betaln(self.alpha1, self.beta1)
        betaln2 = betaln(self.alpha2, self.beta2)
        mean = integrate.dblquad(
            beta_diff.jnt_dst_for_mean,
            -1,
            0,
            lambda x: -x,
            1,
            args=(self.alpha1, self.beta1, self.alpha2, self.beta2, betaln1, betaln2),
        )[0]
        mean += integrate.dblquad(
            beta_diff.jnt_dst_for_mean,
            0,
            1,
            0,
            lambda x: 1 - x,
            args=(self.alpha1, self.beta1, self.alpha2, self.beta2, betaln1, betaln2),
        )[0]
        return mean

    @staticmethod
    def jnt_dst_for_var(s, t, alpha1, beta1, alpha2, beta2, mean, betaln1, betaln2):
        return (t-mean) ** 2 * beta_diff.jnt_dst(s, t, alpha1, beta1, alpha2, beta2, betaln1, betaln2)

    def var(self):
        mean = self.mean()
        betaln1 = betaln(self.alpha1, self.beta1)
        betaln2 = betaln(self.alpha2, self.beta2)
        var = integrate.dblquad(
            beta_diff.jnt_dst_for_var,
            -1,
            0,
            lambda x: -x,
            1,
            args=(self.alpha1, self.beta1, self.alpha2, self.beta2, mean, betaln1, betaln2),
        )[0]
        var += integrate.dblquad(
            beta_diff.jnt_dst_for_var,
            0,
            1,
            0,
            lambda x: 1 - x,
            args=(self.alpha1, self.beta1, self.alpha2, self.beta2, mean, betaln1, betaln2),
        )[0]
        return var

    def std(self):
        return np.sqrt(self.var())