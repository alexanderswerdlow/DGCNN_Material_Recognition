"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
import torch_geometric
import fusion.torch_geometric_extension as ext
import torchvision.models as models
import numpy as np
import ast
import timm

class GraphNetwork(torch.nn.Module):
    def __init__(self, config, nfeat, multigpu=False, default_fnet_widths=[128],
                 default_fnet_llbias=False, default_fnet_tanh=False,
                 default_edge_attr='poscart', default_conv_bias=False):
        super(GraphNetwork, self).__init__()

        self.multigpu = multigpu
        self.devices = []
        self.flow = 'source_to_target'
        nfeat = nfeat
        nEdgeAttr = 0
        self.intermediate = None

        for d, conf in enumerate(config.split(',')):
            fnet_widths = default_fnet_widths
            conf = conf.strip().split('_')
            device = None
            if default_edge_attr is not None:
                edge_attr = [attr.strip() for attr in default_edge_attr.split('-')]
            else:
                edge_attr = []
            fnet_tanh = default_fnet_tanh
            conv_bias = default_conv_bias
            fnet_llbias = default_fnet_llbias

            # Graph Generation
            if conf[0] == 'ggknn':
                if len(conf) < 2:
                    raise RuntimeError("{} Graph Generation layer requires more arguments".format(d))
                neigh = int(conf[1])
                if len(conf) > 2:
                    if conf[2].isdigit():
                        device = conf[2]
                    else:
                        edge_attr = [attr.strip() for attr in conf[2].split('-')]
                        if len(conf) == 4:
                            device = conf[3]
                        elif len(conf) > 4:
                            raise RuntimeError("Invalid parameters in {} ggknn layer".format(d))

                module = ext.GraphReg(knn=True, n_neigh=neigh, edge_attr=edge_attr, self_loop=True, flow=self.flow)
                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            elif conf[0] == 'ggrad':
                if len(conf) < 3:
                    raise RuntimeError("{} Graph Generation layer requires more arguments".format(d))
                rad = float(conf[1])
                neigh = int(conf[2])
                if len(conf) > 3:
                    if conf[3].isdigit():
                        device = conf[3]
                    else:
                        edge_attr = [attr.strip() for attr in conf[3].split('-')]
                        if len(conf) == 5:
                            device = conf[4]
                        elif len(conf) > 5:
                            raise RuntimeError("Invalid parameters in {} ggrad layer".format(d))

                module = ext.GraphReg(knn=False, n_neigh=neigh, rad_neigh=rad,
                                      edge_attr=edge_attr, self_loop=True, flow=self.flow)
                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            elif conf[0] == 'f':
                if len(conf) < 2:
                    raise RuntimeError("{} Fully connected layer requires as argument the output features".format(d))
                nfeato = int(conf[1])
                module = torch.nn.Linear(nfeat, nfeato)
                nfeat = nfeato
                if len(conf) == 3:
                    if conf[2].isdigit():
                        device = conf[2]
                    elif conf[2] == 'cp':
                        torch.nn.init.constant_(module.bias, -np.log(nfeato-1))
                if len(conf) == 4:
                    device = conf[3]
                elif len(conf) > 4:
                    raise RuntimeError("Invalid parameters in {} fully connected layer".format(d))

            elif conf[0] == 'b':
                module = torch.nn.BatchNorm1d(nfeat, affine=True, track_running_stats=True)
                if len(conf) == 2:
                    device = conf[1]
                elif len(conf) > 3:
                    raise RuntimeError("Invalid parameters in {} batchnom layer".format(d))

            elif conf[0] == 'r':
                module = torch.nn.ReLU(True)
                if len(conf) == 2:
                    device = conf[1]
                elif len(conf) > 3:
                    raise RuntimeError("Invalid parameters in {} relu layer".format(d))
            elif conf[0] == 'd':
                if len(conf) < 2:
                    raise RuntimeError(
                        "{} Dropout layer requires as argument the probabity to zeroed an element".format(d))
                prob = float(conf[1])
                module = torch.nn.Dropout(prob, inplace=False)
                if len(conf) == 3:
                    device = conf[2]
                elif len(conf) > 3:
                    raise RuntimeError("Invalid parameters in {} dropout layer".format(d))

            elif conf[0] == 'agc':
                if len(conf) < 2:
                    raise RuntimeError("{} agc layer requires as argument the output features".format(d))
                nfeato = int(conf[1])
                if len(conf) > 2:
                    if conf[2].isdigit():
                        device = conf[2]
                    else:
                        params = [param.strip() for param in conf[2].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                conv_bias = int(p[1])
                            elif param == 'fwidths':
                                fnet_widths = ast.literal_eval(p[1].replace('#', ','))
                            elif param == 'ftanh':
                                fnet_tanh = int(p[1])

                        if len(conf) == 4:
                            device = conf[3]

                        elif len(conf) > 4:
                            raise RuntimeError("Invalid parameters in {} agc layer".format(d))

                module = ext.create_agc(nfeat, nfeato, [nEdgeAttr] + fnet_widths,
                                        fnet_llbias=fnet_llbias,
                                        bias=conv_bias,
                                        fnet_tanh=fnet_tanh,
                                        flow=self.flow)
                nfeat = nfeato

            elif conf[0] == 'multigraphconvknnfeat':
                if len(conf) < 3:
                    raise RuntimeError("{} MUNEGC layer requires as argument the output features".format(d))
                n_neigh = int(conf[1])
                nfeato = int(conf[2])
                if len(conf) > 3:
                    if conf[3].isdigit():
                        device = conf[3]
                    else:
                        params = [param.strip() for param in conf[3].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                conv_bias = int(p[1])
                            elif param == 'fwidths':
                                fnet_widths = ast.literal_eval(p[1].replace('#', ','))
                            elif param == 'ftanh':
                                fnet_tanh = int(p[1])

                        if len(conf) == 5:
                            device = conf[4]

                        elif len(conf) > 5:
                            raise RuntimeError("Invalid parameters in {} MUNEGC layer".format(d))

                module = ext.MUNEGC(nfeat, nfeato,
                                    neighs=n_neigh,
                                    fnetw=fnet_widths,
                                    edge_attr=['posspherical'],
                                    edge_attr_feat=['featureoffsets'],
                                    fnet_llbias=fnet_llbias,
                                    fnet_tanh=fnet_tanh,
                                    bias=conv_bias,
                                    aggr='avg',
                                    flow=self.flow)

                nfeat = nfeato

            # MultiGraphConvolution
            elif conf[0] == 'multigraphconv':
                if len(conf) < 3:
                    raise RuntimeError("{} MUNEGC layer requires as argument the output features".format(d))
                n_neigh = int(conf[1])
                nfeato = int(conf[2])
                if len(conf) > 3:
                    if conf[3].isdigit():
                        device = conf[3]
                    else:
                        params = [param.strip() for param in conf[3].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                conv_bias = int(p[1])
                            elif param == 'fwidths':
                                fnet_widths = ast.literal_eval(p[1].replace('#', ','))
                            elif param == 'ftanh':
                                fnet_tanh = int(p[1])

                        if len(conf) == 5:
                            device = conf[4]

                        elif len(conf) > 5:
                            raise RuntimeError("Invalid parameters in {} MUNEGC layer".format(d))

                module = ext.MUNEGC(nfeat, nfeato,
                                    neighs=n_neigh,
                                    fnetw=fnet_widths,
                                    edge_attr=edge_attr,
                                    fnet_llbias=fnet_llbias,
                                    fnet_tanh=fnet_tanh,
                                    bias=conv_bias,
                                    aggr='avg',
                                    flow=self.flow)

                nfeat = nfeato

            # MultiGraphConvolution
            elif conf[0] == 'multigraphconvmax':
                if len(conf) < 3:
                    raise RuntimeError("{} MUNEGC layer requires as argument the output features".format(d))
                n_neigh = int(conf[1])
                nfeato = int(conf[2])
                if len(conf) > 3:
                    if conf[3].isdigit():
                        device = conf[3]
                    else:
                        params = [param.strip() for param in conf[3].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                conv_bias = int(p[1])
                            elif param == 'fwidths':
                                fnet_widths = ast.literal_eval(p[1].replace('#', ','))
                            elif param == 'ftanh':
                                fnet_tanh = int(p[1])

                        if len(conf) == 5:
                            device = conf[4]

                        elif len(conf) > 5:
                            raise RuntimeError("Invalid parameters in {} MUNEGC layer".format(d))

                module = ext.MUNEGC(nfeat, nfeato,
                                    neighs=n_neigh,
                                    fnetw=fnet_widths,
                                    edge_attr=edge_attr,
                                    fnet_llbias=fnet_llbias,
                                    fnet_tanh=fnet_tanh,
                                    bias=conv_bias,
                                    aggr='max',
                                    flow=self.flow)
                nfeat = nfeato

            # MultiGraphConvolutionGen
            elif conf[0] == 'multigraphconvradbasedgen':
                if len(conf) < 5:
                    raise RuntimeError("{} MUNEGC layer requires as argument the output features".format(d))
                n_neigh = int(conf[1])
                rad_neigh = float(conf[2])
                nfeato = int(conf[4])
                aggr = str(conf[3])
                if len(conf) > 5:
                    if conf[5].isdigit():
                        device = conf[5]
                    else:
                        params = [param.strip() for param in conf[5].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                conv_bias = int(p[1])
                            elif param == 'fwidths':
                                fnet_widths = ast.literal_eval(p[1].replace('#', ','))
                            elif param == 'ftanh':
                                fnet_tanh = int(p[1])

                        if len(conf) == 7:
                            device = conf[6]

                        elif len(conf) > 7:
                            raise RuntimeError("Invalid parameters in {} MUNEGC layer".format(d))

                module = ext.MUNEGC(nfeat, nfeato,
                                    neighs=n_neigh,
                                    rad_neigh=rad_neigh,
                                    fnetw=fnet_widths,
                                    edge_attr=edge_attr,
                                    fnet_llbias=fnet_llbias,
                                    fnet_tanh=fnet_tanh,
                                    bias=conv_bias,
                                    aggr=aggr,
                                    flow=self.flow)
                nfeat = nfeato

            elif conf[0] == 'pvknn':
                if len(conf) < 4:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                nn = int(conf[3])
                if len(conf) > 4:
                    if conf[4].isdigit():
                        device = conf[4]
                    else:
                        edge_attr = [attr.strip() for attr in conf[4].split('-')]
                        if len(conf) == 6:
                            device = conf[5]
                        elif len(conf) > 6:
                            raise RuntimeError("Invalid parameters in {} pool layer".format(d))

                module = ext.VGraphPooling(pradius, aggr=aggr,
                                           neighs=nn, self_loop=True,
                                           edge_attr=edge_attr,
                                           flow=self.flow)

                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            elif conf[0] == 'pvrnn':
                if len(conf) < 5:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                rad_neigh = float(conf[3])
                nn = int(conf[4])

                if len(conf) > 5:
                    if conf[5].isdigit():
                        device = conf[5]
                    else:
                        edge_attr = [attr.strip() for attr in conf[5].split('-')]
                        if len(conf) == 7:
                            device = conf[6]
                        elif len(conf) > 7:
                            raise RuntimeError("Invalid parameters in {} pool layer".format(d))

                module = ext.VGraphPooling(pradius, aggr=aggr,
                                            neighs=nn, rad_neigh=rad_neigh,
                                            self_loop=True, edge_attr=edge_attr,
                                            flow=self.flow)

                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            # voxel  pooling
            elif conf[0] == 'pv':
                if len(conf) < 3:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                if len(conf) == 4:
                    if conf[3].isdigit():
                        device = conf[3]
                module = ext.VPooling(pool_rad=pradius, aggr=aggr)

            # nearest voxel pooling

            elif conf[0] == 'pnv':
                if len(conf) < 3:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                if len(conf) == 4:
                    if conf[3].isdigit():
                        device = conf[3]
                module = ext.NVPooling(pool_rad=pradius, aggr=aggr)

            # KNN pooling layer nearest voxel
            elif conf[0] == 'pnvknn':
                if len(conf) < 4:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                nn = int(conf[3])
                if len(conf) > 4:
                    if conf[4].isdigit():
                        device = conf[4]
                    else:
                        edge_attr = [attr.strip() for attr in conf[4].split('-')]
                        if len(conf) == 6:
                            device = conf[5]
                        elif len(conf) > 6:
                            raise RuntimeError("Invalid parameters in {} pool layer".format(d))

                module = ext.NVGraphPooling(pradius, aggr=aggr,
                                            neighs=nn, self_loop=True,
                                            edge_attr=edge_attr,
                                            flow=self.flow)

                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            # Radius pooling layer nearest voxel
            elif conf[0] == 'pnvrnn':
                if len(conf) < 5:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                rad_neigh = float(conf[3])
                nn = int(conf[4])

                if len(conf) > 5:
                    if conf[5].isdigit():
                        device = conf[5]
                    else:
                        edge_attr = [attr.strip() for attr in conf[5].split('-')]
                        if len(conf) == 7:
                            device = conf[6]
                        elif len(conf) > 7:
                            raise RuntimeError("Invalid parameters in {} pool layer".format(d))
                module = ext.NVGraphPooling(pradius, aggr=aggr,
                                            neighs=nn, rad_neigh=rad_neigh,
                                            self_loop=True, edge_attr=edge_attr,
                                            flow=self.flow)

                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            elif conf[0] == 'gp':
                if len(conf) < 2:
                    raise RuntimeError("Global Pooling Layer needs more arguments")
                aggr = conf[1]
                module = ext.GlobalPCPooling(aggr)

                if len(conf) == 3:
                    device = conf[2]
            # change edge atribs
            elif conf[0] == 'eg':
                if len(conf) > 1:
                    if conf[1].isdigit():
                        device = conf[1]

                    else:
                        edge_attr = [attr.strip() for attr in conf[1].split('-')]
                        if len(conf) == 3:
                            device = conf[2]
                        elif len(conf) > 3:
                            raise RuntimeError("Invalid parameters in {} edge_generation layer".format(d))

                module = ext.GraphReg(knn=None, edge_attr=edge_attr, flow=self.flow)
                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            else:
                raise RuntimeError("{} layer does not exist".format(conf[0]))

            # Adding layer to modules
            if self.multigpu is True:
                if device is None:
                    raise RuntimeError("Multigpu is enabled and layer {} does not have a gpu assigned.".format(d))
                device = 'cuda:{}'.format(id)
                self.devices.append(device)
                module = module.to(device)
            self.add_module(str(d), module)

    def obtain_intermediate(self, layer):

        self.intermediate = layer

    def forward(self, data):
        for i, module in enumerate(self._modules.values()):
            if self.multigpu:
                data = data.to(self.devices[i])
            if type(module) == torch.nn.Linear or \
               type(module) == torch.nn.BatchNorm1d or \
               type(module) == torch.nn.Dropout or \
               type(module) == torch.nn.ReLU:
                if (type(data) == torch_geometric.data.batch.Batch or
                        type(data) == torch_geometric.data.data.Data):

                    data.x = module(data.x)

                elif (type(data) == torch.Tensor):

                    data = module(data)

                else:
                    raise RuntimeError("Unknonw data type in forward time in {} module".format(type(module)))

            elif type(module) == ext.AGC:
                data.x = module(data.x, data.edge_index, data.edge_attr.float())

            elif type(module) == ext.NVGraphPooling or\
                    type(module) == ext.VGraphPooling or\
                    type(module) == ext.VPooling or\
                    type(module) == ext.NVPooling or\
                    type(module) == ext.MUNEGC or\
                    type(module) == ext.GraphReg:

                data = module(data)

            elif type(module) == ext.GlobalPCPooling:
                data = module(data.x, data.batch)

            else:
                raise RuntimeError("Unknown Module in forward time")

            if self.intermediate is not None:
                if self.intermediate == i:
                    return data

        return data

class MultiModalGroupFusion(torch.nn.Module):
    def __init__(self, pool_rad):
        super(MultiModalGroupFusion, self).__init__()
        self.pool_rad = pool_rad

    def forward(self, b1, b2):
        pos = torch.cat([b1.pos, b2.pos], 0)
        batch = torch.cat([b1.batch, b2.batch], 0)

        batch, sorted_indx = torch.sort(batch)
        inv_indx = torch.argsort(sorted_indx)
        pos = pos[sorted_indx, :]

        start = pos.min(dim=0)[0] - self.pool_rad * 0.5
        end = pos.max(dim=0)[0] + self.pool_rad * 0.5

        cluster = torch_geometric.nn.voxel_grid(pos, batch, self.pool_rad, start=start, end=end)
        cluster, perm = consecutive_cluster(cluster)

        superpoint = scatter(pos, cluster, dim=0, reduce='mean')
        new_batch = batch[perm]

        cluster = nearest(pos, superpoint, batch, new_batch)

        cluster, perm = consecutive_cluster(cluster)

        pos = scatter(pos, cluster, dim=0, reduce='mean')
        branch_mask = torch.zeros(batch.size(0)).bool()
        branch_mask[0:b1.batch.size(0)] = 1

        cluster = cluster[inv_indx]

        nVoxels = len(cluster.unique())

        x_b1 = torch.ones(nVoxels, b1.x.shape[1], device=b1.x.device)
        x_b2 = torch.ones(nVoxels, b2.x.shape[1], device=b2.x.device)

        x_b1 = scatter(b1.x, cluster[branch_mask], dim=0, out=x_b1, reduce='mean')
        x_b2 = scatter(b2.x, cluster[~branch_mask], dim=0, out=x_b2, reduce='mean')

        x = torch.cat([x_b1, x_b2], 1)

        batch = batch[perm]

        b1.x = x
        b1.pos = pos
        b1.batch = batch
        b1.edge_attr = None
        b1.edge_index = None

        return b1

    def extra_repr(self):
        s = 'pool_rad: {pool_rad}'
        return s.format(**self.__dict__)


class TwoStreamNetwork(torch.nn.Module):
    def __init__(self, graph_net_conf,
                 features_b1, features_b2, rad_fuse_pool,
                 multigpu,
                 features_proj_b1=64,
                 features_proj_b2=64,
                 proj_b1=True,
                 proj_b2=True):

        super(TwoStreamNetwork, self).__init__()
        self.rad_fuse_pool = float(rad_fuse_pool)
        if proj_b1:
            self.proj_b1 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv1d(
                features_b1, features_proj_b1, kernel_size=1, bias=False))

            features_b1 = features_proj_b1
        else:
            self.proj_b1 = None

        if proj_b2:
            self.proj_b2 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv1d(
                features_b2, features_proj_b2, kernel_size=1, bias=False))
            features_b2 = features_proj_b2
        else:
            self.proj_b2 = None

        self.multimodal_gp_fusion = MultiModalGroupFusion(rad_fuse_pool)
        self.class_network = GraphNetwork(graph_net_conf, features_b1 + features_b2, multigpu)

    def forward(self, data_b1, data_b2):

        batch_b1 = torch_geometric.utils.to_dense_batch(data_b1.x, data_b1.batch)
        batch_b2 = torch_geometric.utils.to_dense_batch(data_b2.x, data_b2.batch)

        x_b1 = batch_b1[0].permute(0, 2, 1)
        x_b2 = batch_b2[0].permute(0, 2, 1)
        if self.proj_b1 is not None:
            x_b1 = self.proj_b1(x_b1)
        if self.proj_b2 is not None:
            x_b2 = self.proj_b2(x_b2)

        data_b1.x = x_b1.permute(0, 2, 1).reshape(-1, x_b1.size(1))[batch_b1[1].view(-1)]
        data_b2.x = x_b2.permute(0, 2, 1).reshape(-1, x_b2.size(1))[batch_b2[1].view(-1)]

        data = self.multimodal_gp_fusion(data_b1, data_b2)
        return self.class_network(data)
