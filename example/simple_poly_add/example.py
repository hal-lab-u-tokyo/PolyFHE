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

    n = 4
    a = [2 * i for i in range(n)]
    b = [3 * i for i in range(n)]
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    compiled.run(a, b, n)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    compile_and_run()
