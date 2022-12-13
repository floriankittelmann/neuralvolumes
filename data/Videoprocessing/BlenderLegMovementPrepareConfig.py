
class BlenderLegMovementPrepareConfig:

    PATH_SLIGHTLY_RAISED = "experiments/blenderLegMovement/data/videos/LegSlightlyRaised"
    PATH_MEDIUM_RAISED = "experiments/blenderLegMovement/data/videos/LegMediumRaised"
    PATH_HIGH_RAISED = "experiments/blenderLegMovement/data/videos/LegHighRaised"

    NOF_FRAMES_OF_THE_VIDEOS = 100

    def __init__(self):
        # train are 32 combinations à 100 pictures -> 32 combinations x 100 pictures x 36 cameras = 115'200 pictures
        self.amplitudes_train: list[str] = [self.PATH_SLIGHTLY_RAISED, self.PATH_MEDIUM_RAISED]
        self.frame_subsampling_train: list[int] = [3, 7, 15, 31]  # Each 3., 7., 15., 31. Frame
        self.phase_shift_subsampling_train: list[int] = [5, 11, 21, 26]  # Start at 5., 11., 21., 26. Frame

        # train are 8 combinations à 100 pictures -> 8 combinations x 100 pictures x 36 cameras = 28'000 pictures
        self.amplitudes_test: list[str] = [self.PATH_SLIGHTLY_RAISED, self.PATH_HIGH_RAISED]
        self.frame_subsampling_test: list[int] = [5, 11]
        self.phase_shift_subsampling_test: list[int] = [7, 13]

    def get_amplitudes_train(self) -> list[str]:
        return self.amplitudes_train

    def get_frame_subsampling_train(self) -> list[int]:
        return self.frame_subsampling_train

    def get_phase_shift_subsampling_train(self) -> list[int]:
        return self.phase_shift_subsampling_train

    def get_amplitudes_test(self) -> list[str]:
        return self.amplitudes_test

    def get_frame_subsampling_test(self) -> list[int]:
        return self.frame_subsampling_test

    def get_phase_shift_subsampling_test(self) -> list[int]:
        return self.phase_shift_subsampling_test
