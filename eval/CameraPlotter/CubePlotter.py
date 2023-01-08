import numpy as np


class CubePlotter:

    def __init__(self):
        self.start = -1  # for all coordinates (x, y, z)
        self.ends = 1  # for all coordinates (x, y, z)

    def draw(self, axis):
        x = np.arange(self.start, self.ends + 0.25, 0.25)
        y = np.arange(self.start, self.ends + 0.25, 0.25)
        x, y = np.meshgrid(x, y)
        z_first = np.ones(x.shape)
        z_second = z_first * -1
        options = {
            'color': 'blue',
            'alpha': 0.3
        }
        # surfaces parallel to x,y axis
        axis.plot_surface(x, y, z_first, **options)
        axis.plot_surface(x, y, z_second, **options)

        # surfaces parallel to x,z axis
        axis.plot_surface(x, z_first, y, **options)
        axis.plot_surface(x, z_second, y, **options)

        # surfaces parallel to y,z axis
        axis.plot_surface(z_first, x, y, **options)
        axis.plot_surface(z_second, x, y, **options)
        return axis
