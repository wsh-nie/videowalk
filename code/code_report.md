# Video Walk代码阅读笔记

[Space-Time Correspondence as a Contrastive Random Walk](https://arxiv.org/abs/2006.14613)

[Resource Code](https://github.com/ajabri/videowalk)

[Comments of Code](https://github.com/wsh-nie/videowalk)



## 数据集处理

### 数据集类

```python
class Kinetics400(VisionDataset):
    def __init__(self, root, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('mp4',), transform=None, cached=None, _precomputed_metadata=None):
        super(Kinetics400, self).__init__(root)
        extensions = extensions

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        # 生成一个元素为元组(path_to_sample, class_idx)的list
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list, # 视频路径
            frames_per_clip, # 每个clips的帧数 
            step_between_clips, # 相邻clips之间的间隔帧数
            frame_rate, # 是否固定视频的fps，固定为多少
            _precomputed_metadata, # 判断是否是第一次处理数据集，第一次处理数据集的时候需要遍历所有数据先获得对应的fps和逐帧时间戳
        )
        self.transform = transform
```

根据`train.py`中传参，可以计算出每个视频的clips数量为`77` clips，计算公式如下：

$$
\frac{10*\text{args.frame_skip} - \text{frames_per_clips}}{1} + 1
$$

```python
    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx) # get subclips 
                success = True
            except:
                print('skipped idx', idx)
                idx = np.random.randint(self.__len__())

        label = self.samples[video_idx][1]
        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label
```

根据`idx`可以获取对应clips，对每个clips进行逐帧操作，先对每帧进行`Crop`和`Resize`操作为固定大小，再分割为多个patches。

### Dataloader

```python
 data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, # shuffle=not args.fast_test,
        sampler=train_sampler, num_workers=args.workers//2,
        pin_memory=True, collate_fn=collate_fn)
```

其中`train_sample=RandomClipSampler(dataset.video_clips, args.clips_per_video)`，即从每个video中随机选择其中的5个clips作为输入参与计算。

输入模型的大小为`[B, T, C*N, Patch_H, Patch_W]`

|                                      | frames_per_clip = 4<br />step_between_clips = 1<br />videos = 6 |
| ------------------------------------ | ------------------------------------------------------------ |
| dataset.video_clips.num_videos()     | 6                                                            |
| dataset.video_clips.num_clips()      | 462                                                          |
| dataset.video_clips.cumulative_sizes | [77, 154, 231, 308, 385, 462]                                |
| dataset.video_clips.video_fps        | [29.97002997002997, 29.97002997002997, 29.97002997002997, 30.0, 30.0, 25.0] |
| data_loader.\_\_len\_\_()            | 15(15*batch_size(2) = 30 = 5\*6)                             |

## Model

### CRW类

```python
class CRW(nn.Module):
    def __init__(self, args, vis=None):
        super(CRW, self).__init__()
        self.args = args

        self.edgedrop_rate = getattr(args, 'dropout', 0)
        self.featdrop_rate = getattr(args, 'featdrop', 0)
        self.temperature = getattr(args, 'temp', getattr(args, 'temperature', 0.07))

        self.encoder = utils.make_encoder(args).to(self.args.device) # 对每个patch提取特征
        self.infer_dims() # set self.enc_hid_dim(特征的通道数) and self.map_scale(降采样倍数)
        self.selfsim_fc = self.make_head(depth=getattr(args, 'head_depth', 0)) # 深度depth +2的全连接层

        self.xent = nn.CrossEntropyLoss(reduction="none")
        self._xent_targets = dict() # 记录每条路径上的计算结果和自监督label

        self.dropout = nn.Dropout(p=self.edgedrop_rate, inplace=False)
        self.featdrop = nn.Dropout(p=self.featdrop_rate, inplace=False)

        self.flip = getattr(args, 'flip', False)
        self.sk_targets = getattr(args, 'sk_targets', False)
        self.vis = vis
```

#### encoder

```python
class From3D(nn.Module):
    ''' Use a 2D convnet as a 3D convnet '''
    def __init__(self, resnet):
        super(From3D, self).__init__()
        self.model = resnet # a resnet network
    
    def forward(self, x):
        N, C, T, h, w = x.shape
        xx = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, h, w) # 把多个patch融入batch_size，以patch大小为输入提取特征
        m = self.model(xx) # [B*N, C, new_h, new_w]

        return m.view(N, T, *m.shape[-3:]).permute(0, 2, 1, 3, 4) # [N, C, T, new_h, new_w]
```

#### selfsim_fc

```python
    def make_head(self, depth=1):
        head = []
        if depth >= 0:
            dims = [self.enc_hid_dim] + [self.enc_hid_dim] * depth + [128]
            for d1, d2 in zip(dims, dims[1:]):
                h = nn.Linear(d1, d2)
                head += [h, nn.ReLU()]
            head = head[:-1]

        return nn.Sequential(*head)
```

### forward

#### Pixels to Nodes

```python
    def pixels_to_nodes(self, x):
        ''' 
            pixel maps -> node embeddings 
            Handles cases where input is a list of patches of images (N>1), or list of whole images (N=1)

            Inputs:
                -- 'x' (B x N x C x T x h x w), batch of images
            Outputs:
                -- 'feats' (B x C x T x N), node embeddings
                -- 'maps'  (B x N x C x T x H x W), node feature maps
        '''
        B, N, C, T, h, w = x.shape
        maps = self.encoder(x.flatten(0, 1)) # shape of `x` is [B*N, C, T, h, w], shape of `maps` is [B*N, enc_hid_dim, T,  Feature_Map_H, Feature_Maps_W]
        H, W = maps.shape[-2:] # size of patch features map

        if self.featdrop_rate > 0:
            maps = self.featdrop(maps)

        # compute node embeddings by spatially pooling node feature maps
        feats = maps.sum(-1).sum(-1) / (H*W) # [B*N, end_hid_dim, T]
        feats = self.selfsim_fc(feats.transpose(-1, -2)).transpose(-1,-2)# [nn.Linear(), nn.ReLU()]
        feats = F.normalize(feats, p=2, dim=1) # normalize at dim=1
    
        feats = feats.view(B, N, feats.shape[1], T).permute(0, 2, 3, 1) # split batch sizes and number of Nodes
        maps  =  maps.view(B, N, *maps.shape[1:])
        """
        `maps` is the result of the input image through the encoder.
        `feats` is the result of a linear and normalize operation after averaging each pixel of the patch feature maps.
        """
        return feats, maps # [1, nd_hid_dim_after_FC, T, N], [B,N,enc_hid_dim, T,  Feature_Map_H, Feature_Maps_W]

```

通过`self.encoder`获得每个patch的feature maps，再将每个patch的feature maps在像素级上求一个平均值，通过一层全连接和一层normalize，将每个node的特征信息压缩为一个数值。

#### Computer Walk

```python
walks = dict()
As = self.affinity(q[:, :, :-1], q[:, :, 1:]) # get the similarity matrix of adjacent nodes, [B,T-1,N,N]
A12s = [self.stoch_mat(As[:, i], do_dropout=True) for i in range(T-1)] # Sequentially split the similarity matrix between frame by frame, [[B,N,N],...] T-1个帧间节点间的相似矩阵
"""
stoch_mat，对输入tensor做一次dropout操作，整体矩阵处于温度常量超参，并在feature map上做一次softmax操作
"""

#################################################### Palindromes
if not self.sk_targets:  
    A21s = [self.stoch_mat(As[:, i].transpose(-1, -2), do_dropout=True) for i in range(T-1)] # Split the similarity matrix between frames in reverse order
    AAs = []
    for i in list(range(1, len(A12s))): # shorter path: 012<->210; 0123<->3210
        g = A12s[:i+1] + A21s[:i+1][::-1]
        aar = aal = g[0]
        for _a in g[1:]:
            aar, aal = aar @ _a, _a @ aal
        AAs.append((f"l{i}", aal) if self.flip else (f"r{i}", aar))
 
    for i, aa in AAs:
        walks[f"cyc {i}"] = [aa, self.xent_targets(aa)] # shorter path similarity matrix and step of `aa`
```

通过`self.affinity`计算逐帧之间节点对应的相似矩阵，然后将帧间的相似矩阵分离为顺序和逆序两个列表，在两个列表上做回文walk，可以得到从起始帧回到起始帧的概率矩阵。

其中`xent_targets`通过自监督产生标签，用于后续计算loss

```python
def xent_targets(self, A):
    B, N = A.shape[:2] # A.shape is [B,N,N]
    key = '%s:%sx%s' % (str(A.device), B,N)

    if key not in self._xent_targets:
        I = torch.arange(A.shape[-1])[None].repeat(B, 1) #[B,N]
        self._xent_targets[key] = I.view(-1).to(A.device) #[B*N]:[0,1,...N-1,0,...]

    return self._xent_targets[key]
```

#### Computer Loss

```python
xents = [torch.tensor([0.]).to(self.args.device)]
diags = dict()

for name, (A, target) in walks.items(): # for each shorter path
    logits = torch.log(A+EPS).flatten(0, -2)
    loss = self.xent(logits, target).mean() # nn.CrossEntropyLoss
    acc = (torch.argmax(logits, dim=-1) == target).float().mean()
    diags.update({f"{H} xent {name}": loss.detach(),
                  f"{H} acc {name}": acc})
    xents += [loss]
```

## 问题

并行问题，无法使用多卡跑起来，最多只能跑2张卡



[^1]:[Pytorch数据加载——Dataset和DataLoader详解](https://blog.csdn.net/loveliuzz/article/details/108756253)