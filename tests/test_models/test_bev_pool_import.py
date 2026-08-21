import pytest

from mmdet3d.ops.bev_pool_v2 import bev_pool as bev_pool_module


def test_missing_bev_pool_extension_reports_build_command(monkeypatch):
    monkeypatch.setattr(bev_pool_module, 'bev_pool_v2_ext', None)
    monkeypatch.setattr(
        bev_pool_module, '_BEV_POOL_IMPORT_ERROR',
        ImportError('extension unavailable'))

    with pytest.raises(RuntimeError, match='build_ext --inplace'):
        bev_pool_module._require_bev_pool_v2_ext()
