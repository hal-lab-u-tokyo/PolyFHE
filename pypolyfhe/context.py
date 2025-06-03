from enum import Enum, auto
from logging import getLogger

# import cupy as cp
import glob
from numba import cuda
import os
import subprocess

logger = getLogger(__name__)


class DeviceType(Enum):
    CPU = auto()
    GPU = auto()


class Device:
    r"""Context-manager that changes the selected device.

    Args:
        device (DeviceType): The type of device to select (CPU or GPU).
    """

    def __init__(self, device: DeviceType):
        self.device_type = device


class CompiledGraph:
    """
    Represents a compiled Poly-Op graph.
    This class is a placeholder for the actual implementation of the compiled graph.
    """

    def __init__(self, name: str):
        self.name = name

    def run(self, build_dir: str = "build"):
        """
        Run the compiled graph.
        This method is a placeholder for the actual implementation of running the compiled graph.
        """
        logger.info(f"Running compiled graph: {self.name}")

        # check if build/polygraph.*.so exists
        so_match = glob.glob(os.path.join(build_dir, "polygraph.*.so"))
        if not so_match:
            logger.error(
                f"Compiled graph not found in {build_dir}. Please compile the graph first."
            )
            return
        logger.info(f"Found compiled graph: {so_match[0]}")

        # import the compiled graph module
        import polygraph

        n = 4
        a = [2 * i for i in range(n)]
        b = [3 * i for i in range(n)]
        print(f"Input a: {a}")
        print(f"Input b: {b}")
        a_gpu = cuda.to_device(a)
        b_gpu = cuda.to_device(b)
        c_gpu = cuda.device_array_like(a_gpu)
        a_gpu_ptr = a_gpu.device_ctypes_pointer.value
        b_gpu_ptr = b_gpu.device_ctypes_pointer.value
        c_gpu_ptr = c_gpu.device_ctypes_pointer.value
        logger.info(
            f"Device pointers: "
            f"a: {a_gpu_ptr:#016x}, "
            f"b: {b_gpu_ptr:#016x}, "
            f"c: {c_gpu_ptr:#016x}"
        )
        polygraph.entry_kernel(a_gpu_ptr, b_gpu_ptr, c_gpu_ptr, n)
        result = c_gpu.copy_to_host()
        print(f"Result: {result}")


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
            CompiledGraph: An instance of CompiledGraph representing the compiled graph.
        """
        logger.info(f"Compiling function {fn.__name__} on device {self.device}.")

        compiled_graph = CompiledGraph(fn.__name__)

        if self.device.device_type == DeviceType.GPU:
            self.build_generated_cu()
        else:
            logger.warning(
                "Compilation for CPU is not implemented yet. Please use GPU for compilation."
            )

        return compiled_graph
