"""Shared DAT test scene: 480 frames of opposing pi/2 rad/s edge rotation.

Reuse the pinned Newton cloth model and simulation loop. Scene inputs are set
before CUDA graph capture, without editing the pristine reference package.
"""
import math
import sys


DEFAULT_FRAMES = 480
ANGULAR_VELOCITY = math.pi / 2
FPS = 60
SUBSTEPS = 10
ITERATIONS = 4
VERTICES = 2500
TRIANGLES = 4802


def make_example_type(original_example, angular_velocity=ANGULAR_VELOCITY):
    if not math.isfinite(angular_velocity) or angular_velocity <= 0:
        raise ValueError("Twist angular velocity must be finite and positive")

    class DatClothTwist(original_example):
        def capture(self):
            actual = (self.model.particle_count, self.model.tri_count,
                      self.fps, self.sim_substeps, self.iterations)
            expected = (VERTICES, TRIANGLES, FPS, SUBSTEPS, ITERATIONS)
            if actual != expected:
                raise RuntimeError(f"DAT twist scene changed: {actual}, expected {expected}")
            # The original scene assigns opposite rotation axes to its edges.
            # Set their common speed before simulate() captures kernel inputs.
            self.rot_angular_velocity = float(angular_velocity)
            return super().capture()

    return DatClothTwist


def run(arguments):
    import newton.examples
    from newton.examples.cloth.example_cloth_twist import Example

    original_arguments = sys.argv
    try:
        sys.argv = ["dat-cloth-twist", *arguments]
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_FRAMES)
        viewer, args = newton.examples.init(parser)
        example = make_example_type(Example)(viewer, args)
        newton.examples.run(example, args)
        return 0
    except SystemExit as result:
        return int(result.code or 0)
    finally:
        sys.argv = original_arguments
