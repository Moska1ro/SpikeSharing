# Spiking pruning
| Model                    | acc    |
| :----------------------- | :----- |
| resnet18                 | 90以上 |
| 直接训练的spike resnet18 | 82     |
| 改进的spike resnet       | 90以上 |
| 直接训练的spike VGG      | 86     |

RepoArch:.<br/>
├─remote_res（远程日志暂存）<br/>
├─results   （归档日志）<br/>
├─tmp       <br/>
├─utils     （依赖）<br/>
│  └─__pycache__<br/>
└─__pycache__（缓存）<br/>
Pretrain param files(.pt) are aviliable at <a href="https://drive.google.com/drive/folders/1aQkOARb6OVNwQGO5KQzV-QJoNhTwKm96?usp=drive_link">here</a>

cp：
+ Notes
  + // 笔记写在这里
+ Todos
  + // Todo写在这里

m：
+ Notes：
  + 提高直接训练的spike resnet acc，有不少文章借鉴，spike residual learning是一个研究方向
  + 当前的研究方向为剪枝，找到适合剪枝的网络是关键
  + 预加载大T训练出的参数，使用小T test，效果下降很少。可以用大T训练，小T剪枝/评估
+ Todos：
  + 尝试小T训练的参数加载到大T中验证
  + 尝试借鉴其他的文章改进spike resnet
  + 构建单独计算适应度的代码

