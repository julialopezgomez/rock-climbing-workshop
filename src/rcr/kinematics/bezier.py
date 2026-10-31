"""Bezier curves in time, and the control points that join two states continuously in position,
velocity and acceleration [C-CROC: Fernbach et al. 2018, https://laas.hal.science/hal-01894869,
eq. 4]. The class is after ndcurves' bezier_curve (Horner evaluation)."""

import numpy as np


class Bezier:
    """c(t) = sum_i B_i^n((t - t_min) / (t_max - t_min)) P_i, t in [t_min, t_max], n + 1 = the
    number of control points."""

    def __init__(self, pointlist, t_min=0.0, t_max=1.0, mult_t=1.0):
        self.dim_ = pointlist[0].shape[0]
        self.T_min_ = t_min
        self.T_max_ = t_max
        self.mult_T_ = mult_t
        self.size_ = len(pointlist)  # number of control points, as in ndcurves
        self.degree_ = self.size_ - 1
        self.control_points_ = pointlist
        if self.size_ < 1 or self.T_max_ <= self.T_min_:
            raise ValueError("Can't create Bezier curve; min bound is higher than max bound.")

    def __call__(self, t):
        if not (self.T_min_ <= t <= self.T_max_):
            raise ValueError("Can't evaluate Bezier curve, time t is out of range")
        if self.size_ == 1:
            return self.mult_T_ * self.control_points_[0]
        else:
            return self.eval_horner(t)

    def eval_horner(self, t):
        u = (t - self.T_min_) / (self.T_max_ - self.T_min_)
        u_op, bc, tn = 1.0 - u, 1, 1
        tmp = self.control_points_[0] * u_op
        for i in range(1, self.degree_):
            tn *= u
            bc *= (self.degree_ - i + 1) / i
            tmp = (tmp + tn * bc * self.control_points_[i]) * u_op
        return (tmp + tn * u * self.control_points_[-1]) * self.mult_T_

    def derivative(self, order):
        if order == 0:
            return self

        derived_wp = []
        for point1, point2 in zip(self.control_points_[:-1], self.control_points_[1:], strict=True):
            derived_wp.append(self.degree_ * (point2 - point1))

        if not derived_wp:
            derived_wp.append(np.zeros(self.dim_))

        deriv = Bezier(
            derived_wp,
            self.T_min_,
            self.T_max_,
            self.mult_T_ * (1.0 / (self.T_max_ - self.T_min_)),
        )
        return deriv.derivative(order - 1)


def connecting(start, end, duration, middle=()):
    """Control points of a Bezier of `duration` from state start = (c, dc, ddc) (position, velocity,
    acceleration) to state end, with the `middle` control points free in between: P0 = c_s,
    P1 = dc_s T / n + P0, P2 = ddc_s T^2 / (n (n - 1)) + 2 P1 - P0, and the same backward from
    P_n = c_g [C-CROC eq. 4]. Degree n = len(middle) + 5."""
    n = len(middle) + 5

    def first_three(c, dc, ddc):
        p1 = dc * duration / n + c
        return [c, p1, ddc * duration**2 / (n * (n - 1)) + 2 * p1 - c]

    c_g, dc_g, ddc_g = end
    return first_three(*start) + list(middle) + first_three(c_g, -dc_g, ddc_g)[::-1]
