# 🧠 CSCI 167 - Introduction to Deep Learning

Welcome to **CSCI 167: Intro to Deep Learning**!  
This repository contains Jupyter notebooks, assignments, and code exercises based on the core concepts from **_Understanding Deep Learning_ (Simon J.D. Prince, MIT Press)** and **_Mastering PyTorch_ (Ashish Ranjan Jha & Gopinath Pillai, Packt Publishing)**.

---

## 📘 Course Overview

This course introduces the foundations of deep learning — both the mathematical principles and the practical implementation.  
By the end of the course, you’ll understand **how and why deep neural networks work**, and be able to build, train, and deploy your own models using **PyTorch**.

### Learning Objectives
- Understand the theoretical basis of deep learning from first principles.
- Implement deep learning architectures using **PyTorch**.
- Explore applications in vision, language, and reinforcement learning.
- Critically analyze ethical and societal implications of AI systems.

---

## 🧩 Repository Structure

| Folder | Description |
|:-------|:-------------|
| `notebooks/` | Core lecture notebooks derived from *Understanding Deep Learning* (UDL). Each notebook focuses on a major concept or architecture. |
| `pytorch_examples/` | Hands-on code implementations adapted from *Mastering PyTorch*. Includes CNNs, Transformers, GANs, RL, and more. |
| `assignments/` | Weekly exercises and projects reinforcing key topics. |
| `datasets/` | Sample datasets used throughout the course (MNIST, CIFAR-10, COCO, etc.). |
| `utils/` | Helper scripts for model training, visualization, and evaluation. |

---

## 🧠 Notebook Index (Based on *Understanding Deep Learning*)

| Notebook | Core Topic |
|:----------|:------------|
| `01_intro.ipynb` | Overview of Deep Learning, Supervised vs Unsupervised Learning |
| `02_supervised_learning.ipynb` | Linear & Logistic Regression, Cost Functions |
| `03_shallow_networks.ipynb` | Neural Network Basics, Universal Approximation Theorem |
| `04_deep_networks.ipynb` | Deep Architectures, Forward Propagation, and Depth Effects |
| `05_loss_functions.ipynb` | Maximum Likelihood, Cross-Entropy, and Custom Losses |
| `06_fitting_models.ipynb` | Gradient Descent, SGD, Adam, and Optimization Strategies |
| `07_backpropagation.ipynb` | Derivatives, Chain Rule, and Backprop Algorithm |
| `08_performance_metrics.ipynb` | Bias-Variance, Overfitting, and Double Descent |
| `09_regularization.ipynb` | L1/L2, Dropout, Early Stopping, and Implicit Regularization |
| `10_convnets.ipynb` | CNNs, Invariance, and Applications to Images |
| `11_resnets.ipynb` | Skip Connections, BatchNorm, and Residual Learning |
| `12_transformers.ipynb` | Self-Attention, BERT, GPT, and Transformer Architectures |
| `13_graph_networks.ipynb` | GNNs, Graph Convolutions, and Applications |
| `14_unsupervised_learning.ipynb` | Clustering, Dimensionality Reduction, and Generative Models |
| `15_gans.ipynb` | GANs, StyleGAN, Conditional GANs, and Stability Techniques |
| `16_flows.ipynb` | Normalizing Flows, Invertible Networks, and Density Estimation |
| `17_vaes.ipynb` | Variational Autoencoders and the ELBO Objective |
| `18_diffusion_models.ipynb` | Denoising Diffusion Probabilistic Models |
| `19_reinforcement_learning.ipynb` | Q-Learning, Policy Gradients, and Actor-Critic Methods |
| `20_theory.ipynb` | Why Deep Learning Works — Generalization and Capacity |
| `21_ethics.ipynb` | AI Ethics, Fairness, and Responsible Research Practices |

---

## 🔥 Implementation Labs (from *Mastering PyTorch*)

| Lab | Description |
|:----|:-------------|
| `cnn_lstm_captioning.ipynb` | Combine CNNs + LSTMs for Image Captioning using COCO |
| `cnn_architectures.ipynb` | Implement LeNet, AlexNet, VGG, and ResNet from scratch |
| `gan_training.ipynb` | Build and train DCGANs for image synthesis |
| `transformer_language_model.ipynb` | Implement a Transformer for text generation |
| `reinforcement_learning_dqn.ipynb` | Build a DQN agent for Atari (Pong) |
| `style_transfer.ipynb` | Implement Neural Style Transfer using pre-trained VGG |
| `explainable_ai.ipynb` | Model interpretability using Captum |
| `pytorch_deployment.ipynb` | Export models with TorchScript and deploy using Flask/TorchServe |

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/csci167-deep-learning.git
   cd csci167-deep-learning
