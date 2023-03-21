import numpy as np
from scipy import integrate
from scipy.special import beta

class beta_diff:
    def __init__(self, alpha1, beta1, alpha2, beta2):
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2

    @staticmethod
    def jnt_dst(s, t, alpha1, beta1, alpha2, beta2):
        return (
            (s + t) ** (alpha1 - 1)
            * (1 - s - t) ** (beta1 - 1)
            * s ** (alpha2 - 1)
            * (1 - s) ** (beta2 - 1)
        )

    def pdf(self, t):
        if t <= -1 or t >= 1:
            return 0
        elif t <= 0:
            return integrate.quad(
                beta_diff.jnt_dst,
                -t,
                1,
                args=(t, self.alpha1, self.beta1, self.alpha2, self.beta2),
            )[0] / (beta(self.alpha1, self.beta1) * beta(self.alpha2, self.beta2))
        else:
            return integrate.quad(
                beta_diff.jnt_dst,
                0,
                1 - t,
                args=(t, self.alpha1, self.beta1, self.alpha2, self.beta2),
            )[0] / (beta(self.alpha1, self.beta1) * beta(self.alpha2, self.beta2))

    def cdf(self, t):
        if t <= -1:
            return 0
        elif t >= 1:
            return 1
        elif t <= 0:
            return integrate.dblquad(
                beta_diff.jnt_dst,
                -1,
                t,
                lambda x: -x,
                1,
                args=(self.alpha1, self.beta1, self.alpha2, self.beta2),
            )[0] / (beta(self.alpha1, self.beta1) * beta(self.alpha2, self.beta2))
        else:
            return (
                integrate.dblquad(
                    beta_diff.jnt_dst,
                    -1,
                    0,
                    lambda x: -x,
                    1,
                    args=(self.alpha1, self.beta1, self.alpha2, self.beta2),
                )[0]
                + integrate.dblquad(
                    beta_diff.jnt_dst,
                    0,
                    t,
                    0,
                    lambda x: 1 - x,
                    args=(self.alpha1, self.beta1, self.alpha2, self.beta2),
                )[0]
            ) / (beta(self.alpha1, self.beta1) * beta(self.alpha2, self.beta2))

    @staticmethod
    def jnt_dst_for_mean(s, t, alpha1, beta1, alpha2, beta2):
        return (
            t
            * (s + t) ** (alpha1 - 1)
            * (1 - s - t) ** (beta1 - 1)
            * s ** (alpha2 - 1)
            * (1 - s) ** (beta2 - 1)
        )

    def mean(self):
        mean = integrate.dblquad(
            beta_diff.jnt_dst_for_mean,
            -1,
            0,
            lambda x: -x,
            1,
            args=(self.alpha1, self.beta1, self.alpha2, self.beta2),
        )[0]
        mean += integrate.dblquad(
            beta_diff.jnt_dst_for_mean,
            0,
            1,
            0,
            lambda x: 1 - x,
            args=(self.alpha1, self.beta1, self.alpha2, self.beta2),
        )[0]
        mean /= beta(self.alpha1, self.beta1) * beta(self.alpha2, self.beta2)
        return mean

    @staticmethod
    def jnt_dst_for_var(s, t, alpha1, beta1, alpha2, beta2, mean):
        return (
            (t - mean) ** 2
            * (s + t) ** (alpha1 - 1)
            * (1 - s - t) ** (beta1 - 1)
            * s ** (alpha2 - 1)
            * (1 - s) ** (beta2 - 1)
        )

    def var(self):
        mean = self.mean()
        var = integrate.dblquad(
            beta_diff.jnt_dst_for_var,
            -1,
            0,
            lambda x: -x,
            1,
            args=(self.alpha1, self.beta1, self.alpha2, self.beta2, mean),
        )[0]
        var += integrate.dblquad(
            beta_diff.jnt_dst_for_var,
            0,
            1,
            0,
            lambda x: 1 - x,
            args=(self.alpha1, self.beta1, self.alpha2, self.beta2, mean),
        )[0]
        var /= beta(self.alpha1, self.beta1) * beta(self.alpha2, self.beta2)
        return var

    def std(self):
        return np.sqrt(self.var())