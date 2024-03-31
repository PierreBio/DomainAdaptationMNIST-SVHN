# Domain Adaptation for MNIST/SVHN
![logo_domain_adaptation](https://github.com/PierreBio/AdaptationDomain/assets/45881846/be12002b-1c55-47da-93a2-9fafe67c0aad)

This project is carried out in the context of the Artificial Intelligence Masters of **TelecomParis**.

<sub>Made with __Python v3.11__</sub>

## Project

Project to train a DNN using MNIST and then apply it on SVHN data using domain adaptation technique.

In pairs, you are tasked with developing a solution to address the challenge of domain adaptation between two distinct image datasets: MNIST and SVHN. Specifically, you will focus on adapting a deep neural network (DNN) trained on the labeled MNIST dataset to perform accurately on the SVHN dataset, for which you are provided with images but no labels during training. Your ultimate goal is to demonstrate effective domain adaptation by achieving high accuracy on the SVHN test set.

Deliverables

- Adapted DNN Model: A deep neural network model that has been successfully adapted from MNIST to perform well on the SVHN dataset.

- Brief Report: A concise report that outlines your approach, methods used for domain adaptation, challenges encountered, and a discussion of the model's performance on the SVHN test set. The report should also include the accuracy metrics achieved on the SVHN test set to demonstrate the effectiveness of your domain adaptation strategy.

## How to setup?

- First, clone the repository:

```
git clone https://github.com/PierreBio/DomainAdaptationMNIST-SVHN.git
```

- Then go to the root of the project:

```
cd DomainAdaptationMNIST-SVHN
```

- Create a virtual environment:

```
py -m venv venv
```

- Activate your environment:

```
.\venv\Scripts\activate
```

- Install requirements:

```
pip install -r requirements.txt
```

## How to launch?

- Once the project is setup, you can launch it using our initial main script (the first time it'll download datasets):

```
py -m src.main
```

## Method

See [our explanations](docs/METHOD.md) about the whole method.

## Results

See [our results](docs/RESULTS.md).

## Ressources

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)
- [Brief Review of Domain Adaptation](https://arxiv.org/pdf/2010.03978.pdf)
