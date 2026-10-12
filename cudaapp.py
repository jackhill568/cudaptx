import cupy as cp

mod = cp.RawModule(path="vec3add.ptx")
kernel = mod.get_function("vec3add")

nx, ny, nz = 100, 100, 1000
n = nx * ny * nz
a = cp.random.randn(n).astype(cp.float32)
b = cp.random.randn(n).astype(cp.float32)
c = cp.zeros(n, dtype=cp.float32)

block = (4, 4, 16)
grid = (-(-nx // block[0]), -(-ny // block[1]), -(-nz // block[2]))

kernel(grid, block, (a, b, c, cp.int32(nx), cp.int32(ny), cp.int32(nz)))
cp.cuda.Device().synchronize()
expected = a + b
assert cp.allclose(c, expected)
print("PTX works!")
