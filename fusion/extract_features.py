class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()

@torch.no_grad()
def get_image_features(model, loader):
    model.eval()

    nLayer = 1
    activations = SaveFeatures(list(model.children())[nLayer][1].sum)

    for i, batch in enumerate(loader, start=0):
        img, gt, files = batch
        img = img.to('cuda:0')
        gt = gt.to('cuda:0')

        _ = model(img)
        features = activations.features
        for i in range(0, features.size(0)):
            feature = features[i, :, :, :]
            feature = feature.view(features.size(1), -1).permute(1, 0)

            # fname_h5_feat2d = os.path.join(path_h5_feat2d, files[i]+".h5")
            # save_h5_features(fname_h5_feat2d, feature.detach().cpu().numpy())

    activations.close()