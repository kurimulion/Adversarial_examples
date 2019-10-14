# Generating Adversarial with Generative Adversarial Networks
While researchers achieved great success in computer vision tasks using CNNs, it’s a known fact that many machine learning algorithms, e.g. CNNs, fully-connected networks, SVMs, etc., are easily fooled[1, 2, 3]. By carefully adding perturbations to input images, one can mislead known models or even unknown models into classifying input into desired classes. Those carefully designed examples are also known as the adversarial examples. Adversarial perturbations are pretty much imperceptible to human eyes, while those perturbations lead models into misclassifying. In this project, we want to utili conditional GANs to generate adversarial examples, which is inspried by [4, 5, 6], evaluate the generated examples against some models, and compare this method with other commonly used methods, e.g. Fast Gradient Sign Method(FGSM)[2], and Projected Gradient Descent(PGD)[3].  
### This project aims to implement arhitecture in [5]
[[1](https://arxiv.org/abs/1312.6199)] Intriguing properties of neural networks, Christian Szegedy


[[2](https://arxiv.org/abs/1412.6572)] Explaining and harnessing adversarial examples, Ian Goodfellow


[[3](https://arxiv.org/abs/1511.07528)] The Limitations of Deep Learning in Adversarial Settings, Nicolas Papernot


[[4](https://arxiv.org/abs/1706.06083)] Towards Deep Learning Models Resistant to Adversarial Attacks, Aleksander Mądry 


[[5](https://arxiv.org/abs/1801.02610)] Generating Adversarial Examples with Adversarial Networks, Chaowei Xiao


[[6](https://arxiv.org/abs/1903.07282)] Generating Adversarial Examples With Conditional Generative Adversarial Net, Ping Yu


[[7](https://arxiv.org/abs/1908.00706)] AdvGAN++ : Harnessing latent layers for adversary generation, Puneet Mangla


