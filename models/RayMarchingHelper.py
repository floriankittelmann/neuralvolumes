from collections.abc import Iterator
import torch


class RayMarchingHelper:

    def __init__(self, pixelcoords, princpt, focal, camrot, campos, dt):
        # NHWC
        raydir = (pixelcoords - princpt[:, None, None, :]) / focal[:, None, None, :]
        raydir = torch.cat([raydir, torch.ones_like(raydir[:, :, :, 0:1])], dim=-1)
        raydir = torch.sum(camrot[:, None, None, :, :] * raydir[:, :, :, :, None], dim=-2)
        self.raydir = raydir / torch.sqrt(torch.sum(raydir ** 2, dim=-1, keepdim=True))
        self.dt = dt

        # compute raymarching starting points
        with torch.no_grad():
            t1 = (-1.0 - campos[:, None, None, :]) / self.raydir
            t2 = (1.0 - campos[:, None, None, :]) / self.raydir
            tmin = torch.max(torch.min(t1[..., 0], t2[..., 0]),
                             torch.max(torch.min(t1[..., 1], t2[..., 1]),
                                       torch.min(t1[..., 2], t2[..., 2])))
            tmax = torch.min(torch.max(t1[..., 0], t2[..., 0]),
                             torch.min(torch.max(t1[..., 1], t2[..., 1]),
                                       torch.max(t1[..., 2], t2[..., 2])))

            intersections = tmin < tmax
            self.t = torch.where(intersections, tmin, torch.zeros_like(tmin)).clamp(min=0.)
            tmin = torch.where(intersections, tmin, torch.zeros_like(tmin))
            self.tmax = torch.where(intersections, tmax, torch.zeros_like(tmin))

        # random starting point
        self.t = self.t - dt * torch.rand_like(self.t)

        self.raypos = campos[:, None, None, :] + self.raydir * self.t[..., None]  # NHWC

    def do_raymarching(self, volsampler, decout, viewtemplate, stepjitter):
        rayrgb = torch.zeros_like(self.raypos.permute(0, 3, 1, 2))  # NCHW
        rayalpha = torch.zeros_like(rayrgb[:, 0:1, :, :])  # NCHW

        # raymarch
        done = torch.zeros_like(self.t).bool()
        while not done.all():
            valid = torch.prod(torch.gt(self.raypos, -1.0) * torch.lt(self.raypos, 1.0), dim=-1).byte()
            validf = valid.float()
            sample_rgb, sample_alpha = volsampler(self.raypos[:, None, :, :, :], **decout, viewtemplate=viewtemplate)

            done, step = self.__calculate_done(done, stepjitter)

            contrib = ((rayalpha + sample_alpha[:, :, 0, :, :] * step[:, None, :, :]).clamp(
                max=1.) - rayalpha) * validf[:, None, :, :]

            rayrgb = rayrgb + sample_rgb[:, :, 0, :, :] * contrib
            rayalpha = rayalpha + contrib

            self.__make_step_for_raypos(step)
            self.t = self.t + step
        return rayrgb, rayalpha

    def __calculate_done(self, done, stepjitter):
        with torch.no_grad():
            step = self.dt * torch.exp(stepjitter * torch.randn_like(self.t))
            done = done | ((self.t + step) >= self.tmax)
        return done, step

    def __make_step_for_raypos(self, step):
        self.raypos = self.raypos + self.raydir * step[:, :, :, None]

    def iterate_raypos(self, nof_max_iterations: int) -> Iterator:
        stepjitter = 0.01
        nof_iterations = 0
        done = torch.zeros_like(self.t).bool()
        while not done.all():
            done, step = self.__calculate_done(done, stepjitter)
            self.__make_step_for_raypos(step)
            nof_iterations = nof_iterations + 1
            yield self.raypos
            if nof_iterations >= nof_max_iterations:
                done = torch.ones_like(self.t).bool()
