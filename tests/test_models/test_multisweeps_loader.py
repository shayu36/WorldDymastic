import mmdet3d.datasets.pipelines.loading as loading_module
from mmdet3d.datasets.pipelines.loading import (
    LoadMultiViewImageFromMultiSweeps)


def _loader():
    loader = object.__new__(LoadMultiViewImageFromMultiSweeps)
    loader.sweeps_num = 4
    loader.test_mode = True
    return loader


def test_single_gpu_fps_loader_falls_back_without_cam_sweeps(monkeypatch):
    loader = _loader()
    monkeypatch.setattr(loading_module, 'get_dist_info', lambda: (0, 1))
    monkeypatch.setattr(
        loader, 'load_offline', lambda results: {'path': 'offline'})
    monkeypatch.setattr(
        loader, 'load_online', lambda results: {'path': 'online'})

    assert loader({'adjacent': []}) == {'path': 'offline'}


def test_single_gpu_fps_loader_uses_online_cam_sweeps(monkeypatch):
    loader = _loader()
    monkeypatch.setattr(loading_module, 'get_dist_info', lambda: (0, 1))
    monkeypatch.setattr(
        loader, 'load_offline', lambda results: {'path': 'offline'})
    monkeypatch.setattr(
        loader, 'load_online', lambda results: {'path': 'online'})

    results = {'cam_sweeps': {'prev': []}}
    assert loader(results) == {'path': 'online'}
