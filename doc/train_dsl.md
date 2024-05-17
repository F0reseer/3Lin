# Train DSL

This DSL is used to specify program for gpt/train. It can be stored in a file or just in a string variable. Using string is more convinient  from IDE. DSL program consists of variable assignments and operations. 

## Variables


## Operations



To train a model you can run gpt/train. Set of operations gpt/train will perform is stored in CONFIG variable. It's basic DSL
 Parameters of the train run and model are specified at the start of gpt/train/main_gpt.cpp.

* SAVE_MODEL – whether store trained model on disk

* START_ITERATION – should we start from scratch or continue from some checkpoint

* MAX_ITERS – number of iterations to train

* TRAIN_ONE_EPOCH – use all train data once in first epoch (was used for some model quality comparison runs)

* EVAL_INTERVAL – interval in iterations to compute train and test logloss

* EVAL_ITERS – number of iterations for train and test logloss compute

* TRAIN_CONFIG – string combining several batch parameters, has form “aNNbXXwYYslideZZ”, omitted parameters have default value

* aNN – accumulate gradients for NN iterations

* bXX – use XX batches per iteration

* wYY – use YY tokens window

* slideZZ – use sliding window attention, full window is YY * ZZ tokens

* DROP_CONFIG – string combining several training parameters, has form “dropXXchYYlrZZtailNN”, omitted parameters have default value

* dropXX – keep XX fraction of tokens, replace others with special ?? token

* chYY – dropout, keep YY parameters intact, zero out the rest (same set for each layer to improve regularization effect)

* lrZZ – use ZZ learning rate

* tailNN – linearly reduce learning rate at training finish, learning rate is reduced from lr at MAX_ITERS * (1 – 1/NN) to 0 at MAX_ITERS

* MODEL_DIMS – string combining model configuration parameters, has form “eNNqXXttYYdZZ”, omitted parameters have default value. Not all model dim parameters combinations are supported, to list them (and add custom) look TSinlgeComputeContext constructor.

* eNN – size of embedding and state vectors

* qXX – size of query and key vectors

* ttYY – size of tensor compression vectors

* dZZ – depth of the model

* MODEL_FLAG - combination of flags

* MODEL_FLAGS – compute logloss on second half of the window (to get more realistic values for small windows)

* MPF_TUNE_FINAL_LAYER – set if we train final layer, random final layer is good enough for sufficiently wide embeddings

* MPF_TUNE_EMBED – set if we train token embeddings, random embeddings are good enough for sufficiently wide embeddings

* MPF_PPM – use PPM

* MPF_GROK_BINARY_OP – use special code in few place to experiment with modulo 97 arithmetic dataset

* MPF_SIM_QUANT_2BIT – experimental, 2-bit model parameters quantization

Load train data section of main.cpp specifies dataset the model will be trained on.

## distributed run

Currently only pow2 number of same config hosts is supported. Work is distributed among worker hosts by main process. To start a worker run gpt/train with -c 1000 argument  (or other port number instead of 10000). To run main process list all worker IPs with ports in XXXX file and run gpt/train with -m argument.

## multiple GPU

To use multiple GPU devices run gpt/train with -d NN argument, where NN is number of GPUs. To use multiple GPUs in distributed run add -d NN argment to main process launch. All hosts will try to use same number of GPUs, heterogeneous configurations are not supported atm.


