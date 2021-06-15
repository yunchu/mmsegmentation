
def propagate_root_dir(cfg, root_dir=None):
    if root_dir is not None:
        cfg.data_root = root_dir

    assert cfg.data_root is not None and cfg.data_root != ''

    cfg.data.train.data_root = cfg.data_root
    cfg.data.val.data_root = cfg.data_root
    cfg.data.test.data_root = cfg.data_root

    return cfg
