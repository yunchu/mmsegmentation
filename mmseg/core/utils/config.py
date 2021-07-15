
def propagate_root_dir(cfg, root_dir=None):
    if root_dir is not None:
        cfg.data_root = root_dir

    assert cfg.data_root is not None and cfg.data_root != ''

    _add2dataset(cfg.data.train, cfg.data_root)
    _add2dataset(cfg.data.val, cfg.data_root)
    _add2dataset(cfg.data.test, cfg.data_root)

    return cfg


def _add2dataset(cfg, root_dir):
    if 'dataset' in cfg:
        cfg.dataset.data_root = root_dir
    else:
        cfg.data_root = root_dir
