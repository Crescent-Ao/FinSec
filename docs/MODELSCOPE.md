# 当前FinSec模型的ModelScope发布

模型地址保持为[aiops11/Qwen3.5-9B-FinSec](https://modelscope.cn/models/aiops11/Qwen3.5-9B-FinSec)。2026年11月13日起，当前版本为`finsec-nf4-v1.0.0`，发布与统一五阶段实验一致的最终FinSec适配器。

## 模型版本与加载关系

- **当前权重**：173,188,512字节的PEFT LoRA适配器，rank16、alpha32、dropout0.05。
- **训练主线**：金融知识SFT → 安全与工具SFT → DPO安全偏好 → 安全对抗SFT及能力回放。
- **评测口径**：同一Qwen3.5-9B基座以NF4双重量化、BF16计算加载，再附加未合并适配器。
- **固定版本**：模型适配器为`finsec-nf4-v1.0.0`；基座的ModelScope及Hugging Face版本、文件校验信息保存在模型仓库的`finsec_release.json`。
- **历史模型**：原两阶段合并模型保留在`legacy-two-stage-20261113`版本，主分支不再提供旧合并权重，避免默认加载错版本。

## 从源码下载

```bash
python -m pip install -r environment/requirements.modelscope.txt
python scripts/modelscope_finsec.py download --output models/FinSec
```

此命令只下载约173MB适配器并验证权重与配置；公共模型无需访问令牌。需要同时下载约19GB的固定基座时，显式指定：

```bash
python scripts/modelscope_finsec.py download \
  --output models/FinSec --base-output models/Qwen3.5-9B
```

基座权重、模型配置与分词文件已核对为原实验版本。下载器会复用SDK缓存；基座下载结束后校验其内容。

## 推理与Agent

在已安装[GPU依赖](../environment/README.md)的环境中，选择自己的空闲GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/modelscope_finsec.py infer \
  --adapter-dir models/FinSec --base-dir models/Qwen3.5-9B \
  --prompt "请解释债券久期与利率变化之间的关系。"
```

加载器显式设置NF4双重量化、BF16计算、非thinking模板与未合并LoRA。需要重复核验基座时加`--verify-base`；每次均验证适配器内容。

接入现有Pi-Agent入口：

```bash
python finsec.py audit --model-dir models/Qwen3.5-9B \
  --adapter models/FinSec --file examples/ecb_demo.py \
  --python /path/to/gpu_env/bin/python --output runs/audit
```

工具执行需要Node和Linux沙箱等依赖，详见[复现指南](REPRODUCE.md)。最终模型的实验值及分母见[RESULTS.md](RESULTS.md)。发布权重经过校验与NF4推理抽检，本次发布没有重新运行整套基准。

## SDK维护方式

本次通过官方`modelscope-hub==0.6.1`的`HubApi`完成认证、历史版本归档、适配器及文档上传、模型介绍更新和固定版本下载。`upload_folder(sync_remote_repo=False)`不会删除本地目录中未提供的远端文件。

迁移时对旧合并权重的清理采用明确的Git提交，历史版本仍可恢复。SDK登录会签发独立Git凭据；API访问令牌与Git凭据不写入公开源码或远端URL。后续文档更新可使用`upload_file`，多文件使用`upload_folder`。官方接口说明见[SDK仓库](https://github.com/modelscope/modelscope_hub)。
