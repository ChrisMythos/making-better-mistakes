import torch.cuda
import torch.nn
from torchvision import models


def init_model_on_gpu(gpus_per_node, opts):
    arch_dict = models.__dict__
    pretrained = False if not hasattr(opts, "pretrained") else opts.pretrained
    distributed = False if not hasattr(opts, "distributed") else opts.distributed
    print("=> using model '{}', pretrained={}".format(opts.arch, pretrained))
    model = arch_dict[opts.arch](pretrained=pretrained)

    if opts.arch == "resnet18":
        feature_dim = 512
    elif opts.arch == "resnet50":
        feature_dim = 2048
    else:
        ValueError("Unknown architecture ", opts.arch)

    # Configure model FC layer based on options
    if opts.devise or opts.barzdenzler:
        if opts.pretrained or opts.pretrained_folder:
            for param in model.parameters():
                if opts.train_backbone_after == 0:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if opts.use_2fc:
            if opts.use_fc_batchnorm:
                model.fc = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.fc_inner_dim, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(opts.fc_inner_dim),
                    torch.nn.Linear(in_features=opts.fc_inner_dim, out_features=opts.embedding_size, bias=True),
                )
            else:
                model.fc = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.fc_inner_dim, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_features=opts.fc_inner_dim, out_features=opts.embedding_size, bias=True),
                )
        else:
            if opts.use_fc_batchnorm:
                model.fc = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(feature_dim), torch.nn.Linear(in_features=feature_dim, out_features=opts.embedding_size, bias=True)
                )
            else:
                model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=feature_dim, out_features=opts.embedding_size, bias=True))
    else:
        model.fc = torch.nn.Sequential(torch.nn.Dropout(opts.dropout), torch.nn.Linear(in_features=feature_dim, out_features=opts.num_classes, bias=True))

    # Check if CUDA is available and properly initialized
    try:
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            print("WARNING: CUDA is not available. Running on CPU.")
            return model  # Return CPU model
    except Exception as e:
        print(f"WARNING: Error checking CUDA availability: {e}. Running on CPU.")
        return model  # Return CPU model

    # Handle GPU device selection more robustly
    if distributed:
        if opts.gpu is not None:
            try:
                # Try newer PyTorch style device handling first
                device = torch.device(f"cuda:{opts.gpu}")
                model.to(device)
                print(f"Using CUDA device {opts.gpu} with newer style device handling")
            except Exception:
                try:
                    # Fall back to the older style
                    torch.cuda.set_device(opts.gpu)
                    model.cuda(opts.gpu)
                    print(f"Using CUDA device {opts.gpu} with older style device handling")
                except Exception as e:
                    print(f"WARNING: Could not set CUDA device {opts.gpu}: {e}. Using CPU instead.")
                    return model
                    
            # When using a single GPU per process and per DistributedDataParallel
            opts.batch_size = int(opts.batch_size / gpus_per_node)
            opts.workers = int(opts.workers / gpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opts.gpu])
        else:
            try:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
            except Exception as e:
                print(f"WARNING: Could not initialize distributed model: {e}. Using CPU instead.")
                return model
    # For single GPU
    elif opts.gpu is not None:
        try:
            # Try newer PyTorch style device handling first
            device = torch.device(f"cuda:{opts.gpu}")
            model = model.to(device)
            print(f"Using CUDA device {opts.gpu} with newer style device handling")
        except Exception:
            try:
                # Fall back to older style
                torch.cuda.set_device(opts.gpu)
                model = model.cuda(opts.gpu)
                print(f"Using CUDA device {opts.gpu} with older style device handling")
            except Exception as e:
                print(f"WARNING: Could not set CUDA device {opts.gpu}: {e}. Using CPU instead.")
    # For DataParallel (multiple GPUs)
    else:
        try:
            model = torch.nn.DataParallel(model).cuda()
        except Exception as e:
            print(f"WARNING: Could not initialize data parallel model: {e}. Using CPU instead.")

    return model
