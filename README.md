<div align="center">

# ORCA

Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering

Aymen Lassoued, Mohamed Ali Souibgui, Yousri Kessentini

**CVPR 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2603.02438-b31b1b.svg)](https://arxiv.org/abs/2603.02438) · [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

</div>

---

## Framework

<p align="center">
  <img src="assets/pipeline.jpg" alt="ORCA Pipeline" width="95%"/>
</p>



## Installation

```bash
git clone https://github.com/AymenLassoued/ORCA.git
cd ORCA
pip install -e .              # core
pip install -e ".[train]"     # + training (Unsloth, trl)
pip install -e ".[vllm]"      # + vLLM acceleration (~5× throughput)
pip install -e ".[all]"       # everything
```

**Requirements**: Python ≥ 3.10 · PyTorch ≥ 2.1 · CUDA ≥ 12.1 · ≥ 48 GB VRAM (for reproducibility, our setup used 4× L4 GPUs)

## Usage

### Training the Router

```bash
agenticvlm-train \
    --config config/default.yaml \
    --train-data data/router_train.csv \
    --image-dir data/images \
    --output-dir outputs/router_v1 \
    --epochs 4 --save-merged
```

### Inference

```bash
agenticvlm-predict \
    --config config/default.yaml \
    --data data/test.csv \
    --image-dir data/images \
    --output results/predictions.csv
```

vLLM-accelerated (~5× throughput via continuous batching and PagedAttention):

```bash
agenticvlm-predict-vllm \
    --config config/default.yaml \
    --data data/test.csv \
    --image-dir data/images \
    --output results/predictions.csv \
    --tensor-parallel 4 --gpu-mem 0.85
```

### Python API

```python
from agenticvlm.models import (
    GLM4VModel, InternVL3Model, Qwen25VLModel,
    Qwen2OCRModel, Qwen3Model, RouterModel,
)
from agenticvlm.pipeline import AgenticVLMPipeline

pipeline = AgenticVLMPipeline(
    thinker=GLM4VModel("path/to/glm-4v-9b"),
    router=RouterModel("path/to/qwen2.5-vl-7b", adapter_path="path/to/adapter"),
    specialist_model=Qwen25VLModel("path/to/qwen2.5-vl-7b"),
    vision_expert=InternVL3Model("path/to/internvl3"),
    ocr_model=Qwen2OCRModel("path/to/qwen2-vl-ocr-2b"),
    debate_llm=Qwen3Model("path/to/qwen3-8b"),
    checker_llm=Qwen3Model("path/to/qwen3-1.7b", enable_thinking=True),
)
pipeline.load_all_models()
result = pipeline.predict("document.png", "What is the total amount?")
print(result.final_answer)
```


## Configuration

All settings are in [`config/default.yaml`](config/default.yaml): model paths, generation parameters, pipeline settings, LoRA hyperparameters, ANLS threshold.

## Project Structure

```
agenticvlm/
├── acceleration/       # vLLM-accelerated inference (VLM + LLM wrappers, CLI)
├── agents/             # Specialist, debate, judge, sanity-checker agents
├── cli/                # predict, train, evaluate entry points
├── data/               # DocVQADataset, label definitions, preprocessing
├── decoding/           # Turbo DFS (recursive DFS + NLL pruning)
├── evaluation/         # ANLS calculator, batch evaluator
├── models/             # BaseVLM wrappers (GLM-4.5V, Qwen2.5-VL, InternVL3, Qwen3, Router)
├── pipeline/           # QuestionRouter, Orchestrator, StressTest, AgenticVLMPipeline
├── prompts/            # Thinker, router, specialist, debate prompt templates
├── training/           # RouterTrainer + vocabulary shrinking
└── utils/              # Text processing, image utils, device, logging
```

## Citation

```bibtex
@inproceedings{lassoued2026orca,
  title={ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering},
  author={Lassoued, Aymen and Souibgui, Mohamed Ali and Kessentini, Yousri},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026},
  eprint={2603.02438},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.02438}
}
```

## License

[CC BY-NC 4.0](LICENSE) — not free for commercial use.