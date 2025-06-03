from logging import getLogger
import os
import subprocess

logger = getLogger(__name__)


class Device:
    r"""Context-manager that changes the selected device.

    Args:
        device_idx (int): device index to select. Negative values are not allowed.
    """

    def __init__(self, device_idx: int):
        if device_idx < 0:
            logger.error("Device index must be a non-negative integer.")
            raise ValueError("Device index must be a non-negative integer.")
        self.device_idx = device_idx


class Context:
    """
    Context class for compiling and running Poly-Op graphs.
    """

    def __init__(self, device: Device):
        self.device = device

    def build_generated_cu(self, build_dir: str = "build"):
        """
        Build the generated CUDA code using cmake

        Args:
            build_dir (str): Directory where the build files are located. Defaults to "build".

        Returns:
            None
        """
        logger.info(f"Building generated CUDA code in directory: {build_dir}")
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run(
            ["cmake", "--build", build_dir, "-j", str(os.cpu_count())],
        )

    def compile(self, fn: callable):
        """
        Compile the given Poly-Op graph.

        Args:
            fn (callable): The function representing the Poly-Op graph to compile.

        Returns:
            None
        """
        logger.info(
            f"Compiling function {fn.__name__} on device {self.device.device_idx}."
        )

        self.build_generated_cu()
