import cuda.tile as ct
import torch


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)

    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    result = a_tile + b_tile

    ct.store(c, index=(pid,), tile=result)


def test():
    total_size = 8192
    tile_size = 1024

    a = torch.randn(total_size, device="cuda")
    b = torch.randn(total_size, device="cuda")
    c = torch.zeros(total_size, device="cuda")

    grid = (ct.cdiv(total_size, tile_size),)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        vector_add,
        (a, b, c, tile_size),
    )

    torch.testing.assert_close(c, a + b)
    # print(f"c: {c}")


if __name__ == "__main__":
    test()
