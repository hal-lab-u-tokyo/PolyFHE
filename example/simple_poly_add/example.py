from pypolyfhe import Context, Device, DeviceType, Poly
import logging

logger = logging.getLogger(__name__)


def compile_and_run():
    logger.info("compileing and running the example...")
    device = Device(DeviceType.GPU)
    context = Context(device)

    def add_poly(input: dict[str, Poly]) -> dict[str, Poly]:
        poly_a = input["a"]
        poly_b = input["b"]
        return {"res": poly_a}

    compiled = context.compile(add_poly)

    compiled.run()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    compile_and_run()
