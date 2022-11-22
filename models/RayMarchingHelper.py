from collections.abc import Iterator
import torch
from models.volsamplers.warpvoxel import VolSampler


class RayMarchingHelper:
    OUTPUT_IMG = 1
    OUTPUT_VOLUME = 2

    def __init__(self,
                 raypos: torch.Tensor,
                 raydir: torch.Tensor,
                 dt: float,
                 t: torch.Tensor,
                 tmax: torch.Tensor,
                 mode: int = OUTPUT_IMG):
        self.raypos = raypos
        self.raydir = raydir
        self.dt = dt
        self.tmax = tmax
        self.t = t
        self.mode = mode

    def do_raymarching(self, volsampler: VolSampler, decout: dict, viewtemplate: bool, stepjitter: float):
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

            if self.mode == self.OUTPUT_IMG:
                rayrgb = rayrgb + sample_rgb[:, :, 0, :, :] * contrib
                rayalpha = rayalpha + contrib
            else:
                rayrgb_tmp = sample_rgb[:, :, 0, :, :] * contrib
                rayalpha_tmp = contrib
                print(rayrgb.size())
                print(rayalpha.size())
                exit()

            self.__make_step_for_raypos(step)
        return rayrgb, rayalpha

    def __calculate_done(self, done, stepjitter):
        with torch.no_grad():
            step = self.dt * torch.exp(stepjitter * torch.randn_like(self.t))
            done = done | ((self.t + step) >= self.tmax)
        return done, step

    def __make_step_for_raypos(self, step):
        self.raypos = self.raypos + self.raydir * step[:, :, :, None]
        self.t = self.t + step

    def iterate_raypos(self) -> Iterator:
        stepjitter = 0.01
        done = torch.zeros_like(self.t).bool()
        while not done.all():
            done, step = self.__calculate_done(done, stepjitter)
            self.__make_step_for_raypos(step)
            yield self.raypos


def init_with_camera_position(pixelcoords, princpt, focal, camrot, campos, dt) -> RayMarchingHelper:
    """ pixelcoords: coordinates x,y of pixels -> example: image width 1024x667 ->
        axis x: values from 0.0 - 1023.0,
        axis y: values from 0.0 - 666.0
    """
    # NHWC

    # Calculates ratio between image width and sensor width (unit doesn't matter because is a ratio)
    raydir = (pixelcoords - princpt[:, None, None, :]) / focal[:, None, None, :]

    # Adds z-axis and fills z-axis with values of 1.0
    raydir = torch.cat([raydir, torch.ones_like(raydir[:, :, :, 0:1])], dim=-1)

    # apply rotation of the raydir according to camrot - it is a transposed rotation matrix
    rotation = camrot[:, None, None, :, :] * raydir[:, :, :, :, None]

    # reduces one axis which was added. why there was added an additional axis?
    raydir = torch.sum(rotation, dim=-2)

    # normalisation of the beam direction
    raydir = raydir / torch.sqrt(torch.sum(raydir ** 2, dim=-1, keepdim=True))

    # compute raymarching starting points
    with torch.no_grad():
        t1 = (-1.0 - campos[:, None, None, :]) / raydir
        t2 = (1.0 - campos[:, None, None, :]) / raydir
        tmin = torch.max(torch.min(t1[..., 0], t2[..., 0]),
                         torch.max(torch.min(t1[..., 1], t2[..., 1]),
                                   torch.min(t1[..., 2], t2[..., 2])))
        tmax = torch.min(torch.max(t1[..., 0], t2[..., 0]),
                         torch.min(torch.max(t1[..., 1], t2[..., 1]),
                                   torch.max(t1[..., 2], t2[..., 2])))

        intersections = tmin < tmax
        t = torch.where(intersections, tmin, torch.zeros_like(tmin)).clamp(min=0.)
        tmin = torch.where(intersections, tmin, torch.zeros_like(tmin))
        tmax = torch.where(intersections, tmax, torch.zeros_like(tmin))

    # random starting point
    t = t - dt * torch.rand_like(t)

    raypos = campos[:, None, None, :] + raydir * t[..., None]  # NHWC
    print(raypos.size())
    print(raydir.size())
    print(type(dt))
    print(t.size())
    print(tmax.size())
    return RayMarchingHelper(raypos, raydir, dt, t, tmax)
