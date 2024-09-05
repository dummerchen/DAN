
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    try:
        import importlib
        M = getattr(importlib.import_module('models.model_' + model.lower(), package=None), 'Model'+model)
    except Exception as e:
        raise NotImplementedError(e, 'Model [{:s}] is not defined.'.format(model))

    m = M(opt)
    return m
