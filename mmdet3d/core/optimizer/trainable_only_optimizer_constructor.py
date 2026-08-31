"""MMCV optimizer constructor that excludes frozen parameters."""

from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner.optimizer.default_constructor import \
    DefaultOptimizerConstructor
from mmcv.utils import build_from_cfg


@OPTIMIZER_BUILDERS.register_module()
class TrainableOnlyOptimizerConstructor(DefaultOptimizerConstructor):
    """Build optimizer groups from parameters with ``requires_grad=True``.

    MMCV's default constructor may retain frozen parameters in optimizer
    groups.  The project training workflow uses optimizer membership as a
    safety boundary for staged/frozen fine-tuning, so this constructor omits
    them entirely and preserves the supported custom LR multipliers.
    """

    _SUPPORTED_PARAMWISE_KEYS = {'custom_keys', 'bypass_duplicate'}

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        super().__init__(optimizer_cfg, paramwise_cfg)
        unsupported = set(self.paramwise_cfg) - self._SUPPORTED_PARAMWISE_KEYS
        if unsupported:
            raise ValueError(
                'TrainableOnlyOptimizerConstructor does not support '
                f'paramwise options: {sorted(unsupported)}')

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        sorted_keys = sorted(sorted(custom_keys), key=len, reverse=True)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)

        params = []
        seen = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                if bypass_duplicate:
                    continue
                raise RuntimeError(
                    'TrainableOnlyOptimizerConstructor found a duplicate '
                    f'trainable parameter: {name}')
            seen.add(param_id)

            param_group = {'params': [param]}
            for key in sorted_keys:
                if key not in name:
                    continue
                lr_mult = custom_keys[key].get('lr_mult', 1.0)
                if self.base_lr is None and lr_mult != 1.0:
                    raise ValueError(
                        f'base lr is required for lr_mult on {key!r}')
                if self.base_lr is not None:
                    param_group['lr'] = self.base_lr * lr_mult
                if self.base_wd is not None:
                    decay_mult = custom_keys[key].get('decay_mult', 1.0)
                    param_group['weight_decay'] = self.base_wd * decay_mult
                break
            params.append(param_group)

        if not params:
            raise RuntimeError(
                'TrainableOnlyOptimizerConstructor found no trainable '
                'parameters')
        optimizer_cfg['params'] = params
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
