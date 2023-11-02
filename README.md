
# AILabs FLaVor
![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)

AILabs **F**ederated **L**earning **a**nd **V**alidation Framew**or**k allows multiple parties to collaboratively build and evaluate AI models without compromising data privacy. **FLaVor** also provides a python package for users to join us, even if your code is implemented in other programming languages. It enables users to make modifications to their program codes by simply adding certain signals and importing/exporting the necessary files, as demonstrated in the examples.

> ***If you combine good FLaVors, model turns into an orchestra.***

## Installation

#### Install stable versions will FL/FV requirements only.

```bash
pip install https://github.com/ailabstw/flavor/archive/refs/heads/release/stable.zip -U && pip install "flavor[fl]"
```

#### Install stable versions will Inference requirements only.

```bash
pip install https://github.com/ailabstw/flavor/archive/refs/heads/release/stable.zip -U && pip install "flavor[infer]"
```

#### Install stable versions will All requirements

```bash
pip install https://github.com/ailabstw/flavor/archive/refs/heads/release/stable.zip -U && pip install "flavor[all]"
```

## Getting Started

 - [Federated Learning Client](examples/hello-fl-client)
 - [Federated Learning Server](examples/hello-fl-server)
 - [Federated Validation](examples/hello-fv)
 - [Inference Model Deployment](examples/hello-inference)

#### Note
1. FLaVor calls the user's code through subprocess, so it allows environment conflicts with the user's training code, such as different versions of Python or conflicting packages. In addition, FLaVor also supports running programs other than Python.
2. The main purpose of the sample code is to guide users to use FLaVor and deploy pre-existing training code on AILabs framework. It is important to note that the intent is not to provide guidance on using the deep learning framework itself. Therefore, direct modification of the sample code is not recommended.


## Asking for help

If you have any questions please:

1. Refer to the examples.
2. Read the [FL doc](https://harmonia.taimedimg.com/flp/documents/fl/2.0/manuals/)/[FV doc](https://harmonia.taimedimg.com/flp/documents/fv/1.0/manuals/).
3. Search through existing Discussions, or add a new question.
