
# Experimental 3Lin Transformer implementation

This repo contains experimental LLM train and inference code written in C++ and CUDA from scratch with concise (~1000 lines) reference CPU implementation. Code uses experiemental Triple-Linear (3Lin) transformer model. [3Lin transformer](doc/model.md)  lacks ReLu nonlinearities and still achieves competative results. CUDA train implementation is optimized for consumer 40* series GPUs. Models can be trained directly in int8 format consuming 1 byte per parameter of GPU memory during training. To train model at scale code supports multiple GPU per host and distributed training. For distributed training code uses regular TCP sockets. To reduce network traffic gradients are packed to 1-bit per parameter.

# Build

To build the the code CUDA v12.3 and C++ compiler are required, msvc for windows,  cmake+clang for Linux. To support cross platform build files generation this repo uses [fo](doc/fo.md), lightweight solution/build files generator. To generate build files you need to compile fo/fo.cpp and run it with two arguments. First argument is root of source tree, second argument is directory to store build files to.

## Windows

```bash
D:\3lin>fo.exe code sln
```

Then open code.sln from d:\3lin\sln\code.sln.

## Linux

To compile 3lin for linux you need to compile fo.cpp, generate CMakeLists.txt file, run cmake, run make.

```bash
~/3lin/fo$ clang++17 fo.cpp -o fo
~/3lin/fo$ cd ..
~/3lin$ ./fo/fo code make.dir
~/3lin$ cd make.dir
~/3lin/make.dir$ cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo
~/3lin/make.dir$ make
```

# Get train data

Examples in the code use [enwik9](https://mattmahoney.net/dc/textdata.html) dataset and its truncacted version enwik8. Also Hugging Face hosted datasets openwebtext, ontocord/CulturaY, danasone/librusec are used in examples. To import them use hf_import_text/import.py.

# Train model

gpt_train is used to train a model. It is controlled by the [train script](/doc/train_script.md). Default train script is stored in [main_gpt.cpp](/code/gpt/train/main_gpt.cpp) CONFIG variable. To load train script from file run gpt_train with '-c script.txt' argument. 

## distributed run

Currently training can be distributed only among pow2 number of worker hosts. 

To start a worker process run gpt_train with '-w 10000' argument. 10000 specifies port number to use.

To run master process call net_train('worker.txt') function in train script. List worker IP addresses in the file provided to net_train().

## multiple GPU

To use multiple GPU devices set DEVICE_COUNT variable in train script to number of GPUs to use. For distributed runs DEVICE_COUNT is applied on each worker, heterogeneous configurations are not supported.

# Download pretrained model

Pretrained xxx on yyyB mostly russian language tokens model with tokenizer can be downloaded from ZZZ.

# Inference test

To try inferencing from the trained model you can use [gpt_infer](/code/gpt/infer). It runs basic http server on 11311 port and allows sampling continuations from the model. Current implementation is slow and designed for demonstration purposes.

# Tokenizers

[Tokenizers](doc/tokenizer.md) are created by [gpt_tokenizer](/code/gpt/tokenizer).

# license

MIT
