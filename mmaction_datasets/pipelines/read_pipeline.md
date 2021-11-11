

@param

```python
self.train_pipeline = [
      dict(type='DecordInit'),# load
      dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1), # load
      dict(type='DecordDecode'), # load
      dict(type='Resize', scale=(-1, 256)), # argumentations
      dict(type='RandomResizedCrop'), # argumentations
      dict(type='Resize', scale=(224, 224), keep_ratio=False), # argumentations
      dict(type='Flip', flip_ratio=0.5), # argumentations
      dict(type='Normalize', **self.img_norm_cfg), # argumentations
      dict(type='FormatShape', input_format='NCTHW'), # formating
      dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]), # formating
      dict(type='ToTensor', keys=['imgs', 'label']) # formating
    ]
```

@dataset

```python
from mmaction.datasets.pipelines import Compose

self.pipeline = Compose(pipeline)
'''
  Compose(
      DecordInit(io_backend=disk, num_threads=1)
      SampleFrames(clip_len=32, frame_interval=2, num_clips=1, temporal_jitter=False, twice_sample=False, out_of_bound_opt=loop, test_mode=False)
      DecordDecode(mode=accurate)
      Resize(scale=(inf, 256), keep_ratio=True, interpolation=bilinear, lazy=False)
      RandomResizedCrop(area_range=(0.08, 1.0), aspect_ratio_range=(0.75, 1.3333333333333333), lazy=False)
      Resize(scale=(224, 224), keep_ratio=False, interpolation=bilinear, lazy=False)
      Flip(flip_ratio=0.5, direction=horizontal, flip_label_map=None, lazy=False)
      Normalize(mean=[123.675 116.28  103.53 ], std=[58.395 57.12  57.375], to_bgr=False, adjust_magnitude=False)
      FormatShape(input_format='NCTHW')
      Collect(keys=['imgs', 'label'], meta_keys=[], nested=False)
      ToTensor(keys=['imgs', 'label'])
  )
'''
```

@mmaction.datasets.pipeline.compose

```python
from collections.abc import Sequence

from mmcv.utils import build_from_cfg
import ..build

'''
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
BLENDINGS = Registry('blending')
'''

@PIPELINES.register_module()
class Compose: # Registry('pipeline').register_module(Compose)
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

```

`__call__()`[^1]将对象当作函数调用，`__repr__()`[^2]用于实现对象输出格式

装饰器[^3]

@mmcv.utils.registry

```python
def build_from_cfg(cfg, registry, default_args=None): # (transform:dict, PIPELINES: Registry)
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')
```

## 详解PIPELINE[^5]各个处理实现

传入参数`results`有如下`key`：`filename, modality, start_index`

* DecordInit

  通过传参数`results['filename']`使用[`decord`](https://github.com/dmlc/decord)初始化一个`video_reader`，存于`results['video_reader’]`，并获得总帧数`results['total_frames']`

* SampleFrames

```python
def __init__(self, 
             clip_len, #总帧数
             frame_interval=1, #相邻采样帧的时间间隔。
             num_clips=1, #被采样的clips数量
             temporal_jitter=False, #是否应用时间抖动
             twice_sample=False, #设True则采样带有和不带有固定移位的帧，用于TSM模型的测试中
             out_of_bound_opt='loop',#处理越界帧的方法
             test_mode=False,# 是否是val和test数据集
             start_index=None,# 弃用
             frame_uniform=False):#True专用于sthv2数据集采样策略
```

根据需要被采样的帧数以及clip数量获取采样的帧index，存于`results['frame_inds']`，此外保存`results['clip_len']`、`results['frame_interval']`、`results['num_clips']`

* DecordDecode

从`results['video_reader']`中读出对应`results['frame_inds']`的图像，并按照顺序存于`results['imgs']`，同时保存图像原始尺寸和当前尺寸于`results['original_shape']`，`results['img_shape']`

* Resize

```python
def __init__(self,
             scale,# 如果 keep_ratio 为 True，则用作缩放因子或最大尺寸：如果它是一个浮点数，图像将按这个因子重新缩放，否则如果它是 2 个整数的元组，图像将在比例范围内重新缩放尽可能大。否则，它作为输出大小的 (w, h)。
             keep_ratio=True,# 如果设置为 True，图像将在不改变纵横比的情况下调整大小。 否则，它会将图像调整为给定的大小。 默认值：真。
             interpolation='bilinear',# "nearest" | "bilinear". Default: "bilinear".
             lazy=False):
```

`scale=(-1, 256)`将图像最小一维resize为256，另一维度同比例缩放；`scale=(224,224)`则将图像采样为制定大小

更新`img_shape`和`img`，同时保存是否横纵保存同比例缩放以及各维度缩放因子`results['keep_ratio']`，`results['scale_factor']`。

* RandomResizedCrop

```python
def __init__(self,
             area_range=(0.08, 1.0),#候选区域缩放输出裁剪图像的范围
             aspect_ratio_range=(3 / 4, 4 / 3),#候选纵横比范围
             lazy=False)
```

随机采样和缩放；`results['crop_quadruple’]`保存x和y方向的偏移率以及宽高缩放比（用于对`gt_bboxes`的更新），`results['crop_bbox’]`保存采样图像在原图的左上角和右下角位置，更新`results['img_shape’]`和`results['img']`

* Flip

```python
"""
以概率翻转输入图像。
     以特定方向反转给定 imgs 中元素的顺序。
     imgs 的形状被保留，但元素被重新排序。
"""
def __init__(self,
             flip_ratio=0.5, # Probability of implementing flip.
             direction='horizontal',# Flip imgs horizontally or vertically
             flip_label_map=None,# (Dict[int, int] | None): Transform the label of the flipped image with the specific label.
             left_kp=None,
             right_kp=None,
             lazy=False):
```

`results['flip']`保存是否进行翻转，`results['flip_direction']`保存横纵向翻转方向，使用`cv.flip`对`results['img']`进行翻转

* Normalize

Normalize images with the given mean and std value.

* FormatShape

```python
def __init__(self, 
						 input_format, # Define the final imgs format.['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']
             collapse=False):
```

`N`是`results['num_clips'] * N_crops`，`C`是`RGB`通道，`T`是每个`clips`的帧数，`HW`分别是图像的高宽

* Collect

从加载器收集与特定任务相关的数据。根据输入提取需要的信息

* ToTensor

Convert some values in results dict to `torch.Tensor` type in data loader pipeline.

* CenterCrop
从中心剪裁制定大小图像

* ThreeCrop

沿着较短的边以相等的间隔将图像平均地裁剪成三个裁剪。

-----
参考链接
[^1]:[`__call__`函数](https://blog.csdn.net/Yaokai_AssultMaster/article/details/70256621)

[^2]:[`__repr__`函数](https://zhuanlan.zhihu.com/p/80911576)

[^3]: [理解python 装饰器](https://www.zhihu.com/question/26930016)

[^4]: [装饰器`@property`](https://www.liaoxuefeng.com/wiki/1016959663602400/1017502538658208)：避免直接暴露参数导致随意修改参数，该属性通过”getter”和”setter"方法来实现，但可直接对属性名操作。

[^5]: [mmaction.dataset.pipeline](https://mmaction2.readthedocs.io/zh_CN/latest/api.html#module-mmaction.datasets.pipelines)


