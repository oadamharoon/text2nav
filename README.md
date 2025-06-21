# Can Pretrained Vision-Language Embeddings Alone Guide Robot Navigation?

[![arXiv](https://img.shields.io/badge/arXiv-2506.14507-b31b1b.svg)](https://arxiv.org/abs/2506.14507)
[![RSS 2025](https://img.shields.io/badge/RSS%202025-Workshop-blue)](https://sites.google.com/brown.edu/fm4roboplan/home)
[![Python](https://img.shields.io/badge/Python-76.3%25-blue)](https://github.com/oadamharoon/text2nav)
[![Jupyter](https://img.shields.io/badge/Jupyter%20Notebook-22.7%25-orange)](https://github.com/oadamharoon/text2nav)

**Repository: text2nav**

*Accepted to Robotics: Science and Systems (RSS) 2025 Workshop on Robot Planning in the Era of Foundation Models (FM4RoboPlan)*

## 📝 Overview

This repository contains the implementation for our research investigating whether frozen vision-language model embeddings can guide robot navigation without fine-tuning or specialized architectures. We present a minimalist framework that achieves **74% success rate** in language-guided navigation using only pretrained SigLIP embeddings.

## 🎯 Key Findings

- 🎯 **74% success rate** using frozen VLM embeddings alone (vs 100% privileged expert)
- 🔍 **3.2x longer paths** compared to privileged expert, revealing efficiency limitations
- 📊 **SigLIP outperforms CLIP and ViLT** for navigation tasks (74% vs 62% vs 40%)
- ⚖️ Clear **performance-complexity tradeoffs** for resource-constrained applications
- 🧠 Strong semantic grounding but limitations in spatial reasoning and planning

## 🚀 Method

Our approach consists of two phases:

1. **Expert Demonstration Phase**: Train a privileged policy with full state access using PPO
2. **Behavioral Cloning Phase**: Distill expert knowledge into a policy using only frozen VLM embeddings

The key insight is using frozen vision-language embeddings as drop-in representations without any fine-tuning, providing an empirical baseline for understanding foundation model capabilities in embodied tasks.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NVIDIA Isaac Sim/Isaac Lab
- PyTorch
- CUDA-compatible GPU

### Setup
```bash
git clone https://github.com/oadamharoon/text2nav.git
cd text2nav

# Install dependencies
pip install torch torchvision
pip install transformers
pip install numpy matplotlib
pip install gymnasium

# For Isaac Lab simulation (follow official installation guide)
# https://isaac-sim.github.io/IsaacLab/
```

## 📁 Repository Structure

```
text2nav/
├── README.md              # This documentation
├── CITATION.cff           # Citation information
├── LICENSE                # MIT License
├── IsaacLab/              # Isaac Lab simulation environment setup
├── embeddings/            # Vision-language embedding generation
├── rl/                    # Reinforcement learning expert training
├── generate_embeddings.ipynb    # Generate VLM embeddings from demonstrations
├── revised_gen_embed.ipynb      # Revised embedding generation
├── train_offline.py             # Behavioral cloning training script
├── offlin_train.py              # Alternative offline training
├── bc_model.pt                  # Trained behavioral cloning model
├── td3_bc_model.pt            # TD3+BC baseline model
├── habitat_test.ipynb         # Testing and evaluation notebook
└── replay_buffer.py           # Data handling utilities
```

## 🎮 Usage

### 1. Expert Demonstration Collection
```bash
cd rl/
python train_expert.py --env isaac_sim --num_episodes 500
```

### 2. Generate VLM Embeddings
```bash
jupyter notebook generate_embeddings.ipynb
```

### 3. Train Navigation Policy
```bash
python train_offline.py --model siglip --embedding_dim 1152 --batch_size 32
```

### 4. Evaluate Policy
```bash
jupyter notebook habitat_test.ipynb
```

## 📊 Results

| Model | Success Rate (%) | Avg Steps | Efficiency |
|-------|------------------|-----------|------------|
| Expert (πβ) | 100.0 | 113.97 | 1.0× |
| SigLIP | 74.0 | 369.4 | 3.2× |
| CLIP | 62.0 | 417.6 | 3.7× |
| ViLT | 40.0 | 472.0 | 4.1× |

## 🔬 Experimental Setup

- **Environment**: 3m × 3m arena in NVIDIA Isaac Sim
- **Robot**: NVIDIA JetBot with RGB camera (256×256)
- **Task**: Navigate to colored spheres based on language instructions
- **Targets**: 5 colored spheres (red, green, blue, yellow, pink)
- **Success Criteria**: Reach within 0.1m of correct target

## 💡 Key Insights

1. **Semantic Grounding**: Pretrained VLMs excel at connecting language descriptions to visual observations
2. **Spatial Limitations**: Frozen embeddings struggle with long-horizon planning and spatial reasoning
3. **Prompt Engineering**: Including relative spatial cues significantly improves performance
4. **Embedding Dimensionality**: Higher-dimensional embeddings (SigLIP: 1152D) outperform lower-dimensional ones

## 🔮 Future Work

- Hybrid architectures combining frozen embeddings with lightweight spatial memory
- Data-efficient adaptation techniques to bridge the efficiency gap
- Testing in more complex environments with obstacles and natural language variation
- Integration with world models for better spatial reasoning

## 📚 Citation

```bibtex
@misc{subedi2025pretrainedvisionlanguageembeddingsguide,
      title={Can Pretrained Vision-Language Embeddings Alone Guide Robot Navigation?}, 
      author={Nitesh Subedi and Adam Haroon and Shreyan Ganguly and Samuel T. K. Tetteh and Prajwal Koirala and Cody Fleming and Soumik Sarkar},
      year={2025},
      eprint={2506.14507},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.14507}, 
}
```

## 🙏 Acknowledgments

This work is funded by NSF-USDA COALESCE grant #2021-67021-34418. Special thanks to the Iowa State University Mechanical Engineering Department for their support.

## 👥 Contributors

- [Nitesh Subedi](https://github.com/nitesh-subedi)* (Iowa State University)
- [Adam Haroon](https://github.com/oadamharoon)* (Iowa State University)  
- [Shreyan Ganguly](https://github.com/tre3x) (Iowa State University)
- [Samuel T.K. Tetteh](https://github.com/samtett) (Iowa State University)
- [Prajwal Koirala](https://github.com/prajwalkoirala) (Iowa State University)
- [Cody Fleming](https://github.com/codyfleming) (Iowa State University)
- [Soumik Sarkar](https://github.com/soumiks) (Iowa State University)

*Equal contribution

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Paper (arXiv)](https://arxiv.org/abs/2506.14507)
- [RSS 2025 FM4RoboPlan Workshop](https://sites.google.com/brown.edu/fm4roboplan/home)
- [Iowa State University](https://www.iastate.edu/)

---

*For questions or issues, please open a GitHub issue or contact the authors.*
