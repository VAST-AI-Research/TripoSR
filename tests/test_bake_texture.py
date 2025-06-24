import numpy as np, torch
from tsr import bake_texture as bt

def test_dtype_cast_cpu():
    fake_mesh      = type("M", (), {})()      # stub so we don’t import xatlas
    fake_mesh.vertices = np.zeros((4, 3), np.float32)
    fake_mesh.faces    = np.zeros((2, 3), np.int32)

    # 4×4 RGBA image full of zeros – dtype float32
    dummy_tex = np.zeros((4, 4, 4), np.float32)

    class DummyRenderer:
        def query_triplane(self, dec, pos, code):
            assert pos.dtype == torch.float32   # key assertion
            return {"color": torch.zeros_like(pos[:, :3])}

    class DummyModel:
        renderer = DummyRenderer()
        decoder  = None

    out = bt.positions_to_colors(DummyModel(), torch.zeros(1,1), dummy_tex, 4)
    assert out.shape == (4, 4, 4)
