from eval.CoordinateSystem import CoordinateSystem
from scipy.spatial.transform import Rotation


class GlobalCoordinateSystem(CoordinateSystem):
    def __init__(self):
        rot_nothing = Rotation.from_matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
        )
        super().__init__(0.0, 0.0, 0.0, rot_nothing)
