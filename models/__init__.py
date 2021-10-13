from models.jtfn import JTFN

import torch

def create_model(
    args
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parametes

    """
    archs = [JTFN]
    archs_dict = {a.__name__.lower(): a for a in archs}
    arch = args['architecture']
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError("Wrong architecture type `{}`. Available options are: {}".format(
            arch, list(archs_dict.keys()),
        ))
    if arch.lower() == 'jtfn':
        return model_class(args.backbone, args.use_gau, args.use_fim, args.up, args.classes, args.steps, reduce_dim=args.reduce_dim)
    else:
        raise RuntimeError('No implementation: ', arch.lower())