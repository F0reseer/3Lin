# Train Script

Training model with [train_gpt](/code/gpt/train) is controlled by the script written on simple DSL. Script consists of variable assignments and operations. Script looks like this:

```
VARIABLE = 100 # comment
OPERATION(12, 'asdf')
```

## Variables

* **MAX_ITERS = 50000**
Set train iterations count

* **DEVICE_COUNT = 4**
Set number of GPUs to use, default is 1

* **EVAL_INTERVAL = 1000**
Train&test log loss is computed every EVAL_INTERVAL iteration. 

* **EVAL_ITERS = 20**
Number of iterations that are used to compute train&test log loss. Loss is computed on fixed set of batches, this introduces some bias but greatly reduces variance, which is better for comparing different model and training method changes.

* **TEST_FRACTION = 0.01**
Fraction of dataset used for test, default is 0.05

* **MODEL_DIMS = 'e512tt128d39'**
String, containing model dimensions, see below

* **TRAIN_CONFIG = 'b16ww1024'**
String, containing batch parameters, see below

* **DROP_CONFIG = 'drop1ch1'**
String containing dropout and learning rate parameters, see below

* **SAVE_MODEL = false**
If SAVE_MODEL is set model is saved on disk every EVAL_INTERVAL iterations, otherwise only achieved loss is displayed. Default is yes.

* **USE_PPM = true**
If USE_PPM is set [PPM](doc/ppm.md) feature is added to model input. Default is no.

## Model operations

* **create_model(flag1, flag2..)**
Create new model. Takes list of MPF_* flags on input. See MPF_* flags description below

* **load_model('filename.bin')**
Load model from binary file. Model is saved to binary files during training if SAVE_MODEL is set.

* **load_checkpoint(N)**
N - number of iteration to load model from. Can be used to continue aborted for some reason training run.

## Tokenizer operations

* **set_vocab_size(N)**
N - token count, used with load_tokenized_* functions. N = 50257 for gpt2 tokenizer.

* **set_doc_start_token(N)**
N - token id of doc delimiting token. Can be used with load_tokenized_* functions.

* **load_tokenizer('filename.bin')**
Load tokenizer from binary file. [Tokenizer](doc/tokenizer) binary file can be created with [gpt_tokenizer](code/gpt/tokenizer).

* **make_byte_tokenizer()**
Create byte tokenizer. Generates one token per bytes, uses 256 different tokens, one for each byte value. 

## Dataset operations

* **make_char_dataset('shakespear.txt')**
Loads single text file. Creates tokenizer out of used bytes in this text file. Splits file into two parts, first part is used  for train, second for test. 

* **load_tokenized_train('train.bin')**
Load tokenized dataset and add it to train set. Argument specifies binary file with sequence of tokens. Each token is stored as ui16.
 
* **load_tokenized_test('test.bin')**
Load tokenized dataset and it it to test set.

* **load_text('doc1.txt')**
Load text file, tokenize it with selected tokenizer and add to dataset. Code assumes utf8 encoding of the text.

* **load_folder('cultura')**
Load all files in the specified folder and add them to dataset. Each file is considered a text document.

* **load_docset('cultura/2.bin')**
Load document pack and add each document to dataset. Document packs can be created with [hf import](code/hf_import_text). Document pack is  a binary file consisting of serialized documents. Each document has 4 byte header with document length followed by utf8 encoded text of the document.

* **index_docset_folder('cultura')**
Tokenize and create PPM feature for all document packs in the specified folder. Stores result to index.bin and index_hdr.bin files. Can be used to preprocess large datasets once and then use them to train models.

* **load_indexed_docset_folder('cultura')**
Load tokenized with inde_docset_folder() documents and add them to dataset. The only way to work with document collections which do not fit into RAM is to index them with index_docset_folder() and then load them with load_indexed_docset_folder().

* **save_dataset('dataset.bin')**
Save dataset to binary file. Can be used to avoid tokenization on each script execution.

* **load_dataset('dataset.bin')**
Load dataset saved with save_dataset()

## Process operations

* **train()**
Train model on local host. MAX_ITERS iterations are performed. Train and test log loss scores are reported along the training process.

* **net_train('workers.txt')**
Perform distributed model train. List of worker IP address is loaded from text file provided. One IP address on each line is expected. 

* **compute_exact_test(Ncheckpoint, Navrg)**
Load model from Ncheckpoint iteration. If Navrg is non zero then load all models in [Ncheckpoint - Navrg; Ncheckpoint] range and average them. Sample random test batches, report average score over all sampled batches.

## TRAIN_CONFIG

**TRAIN_CONFIG** – string combining several batch parameters, it has form “aNNbXXwwYYwZZslideSS”. Omitted parameters have default value.

* aNN – accumulate gradients for NN iterations, default is 1

* bXX – use XX batches per iteration

* wwYY – use YY tokens wide window

* wZZ - use ZZ tokens normal window, default is 64

* slideSS – use sliding window attention, full window is YY * ZZ tokens, default is 1

## MODEL_DIMS

**MODEL_DIMS** – string combining model configuration parameters, it has form “eNNqXXttYYdZZ”. Omitted parameters have default value. Not all model dim parameters combinations are supported, to list them (and add custom) look [TSinlgeComputeContext constructor](code/gpt/model/gpt_cuda.cu).

* eNN – size of embedding and state vectors

* qXX – size of query and key vectors, default is 128

* ttYY – size of tensor compression vectors, default is 64

* dZZ – depth of the model


## DROP_CONFIG

**DROP_CONFIG** – string combining several training parameters, it has form “dropXXchYYlrZZtailNN”. Omitted parameters have default value.

* dropXX – keep XX fraction of tokens, replace others with special ?? token

* chYY – dropout, keep YY parameters intact, zero out the rest (same set for each layer to improve regularization effect)

* lrZZ – use ZZ learning rate

* tailNN – linearly reduce learning rate at training finish, learning rate is reduced from lr at MAX_ITERS * (1 – 1/NN) to 0 at MAX_ITERS

## create_model() flags

* MPF_TAIL_LOSS – compute logloss on second half of the window (to get more realistic values for small windows)

* MPF_TUNE_FINAL_LAYER – set if we train final layer, random final layer is good enough for sufficiently wide embeddings

* MPF_TUNE_EMBED – set if we train token embeddings, random embeddings are good enough for sufficiently wide embeddings

* MPF_GROK_BINARY_OP – use special code in few place to experiment with modulo 97 arithmetic dataset

* MPF_SIM_QUANT_2BIT – experimental, 2-bit model parameters quantization

## Script examples

There are few example scripts in [main_gpt.cpp](code/gpt/train/main_gpt.cpp). To load train script from file run gpt_train with '-c script.txt' argument. Shortest valid train script:

```
make_char_dataset('shakespear.txt')
create_model(MPF_TAIL_LOSS)
train()
```

