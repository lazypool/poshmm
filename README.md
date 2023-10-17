# Part-of-Speech based on HMM

A simple Part-of-Speech tagging implemented by python, which is based on the `Hidden Markov Model`.

What this program contributed:

- A simple solution for POS tagging.
- Implementation and application of HMM.
- A very cool self-made text processing tool.

## Introduction

### Workflow

- **September 20th.** Get the corpus and completed the data cleaning.
- **September 21st.** Impletement the main function of the 3 matrix, containing 'read', 'save', 'normalize'.
- **September 22nd.** Use the multiprocessing to accelerate the training. Do a little change to the program's architecture.
- **September 23rd.** Implement the predict methods.
- **September 30th.** Build the framework of test function. Refactored the structure of corpus.
- **October 1st.** Completed the main function. Change the program's architecture a little.
- **October 10th.** Do a series of debug and experiments. Obtain the final model with the highest accuracy. Provide the example input.
- **October 12th.** Fine-tuning the program. Include the hyperparameters.
- **October 15th.** Test again and again to increase the accuracy.

### Highlights

- Train on a medium-sized dataset(2329263 lines, 531 MB) and obtain great performance(91.94% acc).
- Implementing parallel operations, the training speed is increased by about 40%(Train 25s, Test 13s).
- Conduct detailed analysis of error model-predicted samples to perfect the model.

### Examples

The useage examples for the input are saved as the 'example.txt' file.

The example.txt provide 10 middle-length sentences.

You can use the example input with the command:

```bash
python main.py < example.txt
```

After enter this, you should be able to see how the program works.

## Corpus

### Overview

All the corpus are Chinese. They are found at [liwenzhu's repositories](https://github.com/liwenzhu/corpusZh). Thanks for this guy's hard work.

The corpus contains 35 parts of speech in total, a total of 2329263 lines, and 531 MB.

| 代码 |   意义   | 代码 |   意义   | 代码 |   意义   | 代码 |     意义     | 代码 |      意义      |
| :--: | :------: | :--: | :------: | :--: | :------: | :--: | :----------: | :--: | :------------: |
|  n   | 普通名词 |  nt  | 时间名词 |  nd  | 方位名词 |  nl  |   处所名词   |  nh  |      人名      |
| nhf  |    姓    | nhs  |    名    |  ns  |   地名   |  nn  |     族名     |  ni  |     机构名     |
|  nz  | 其他专名 |  v   |   动词   |  vd  | 趋向动词 |  vl  |   联系动词   |  vu  |    能愿动词    |
|  a   |  形容词  |  f   |  区别词  |  m   |   数词   |  q   |     量词     |  d   |      副词      |
|  r   |   代词   |  p   |   介词   |  c   |   连词   |  u   |     助词     |  e   |      叹词      |
|  o   |  拟声词  |  i   |  习用语  |  j   |  缩略语  |  h   |   前接成分   |  k   |    后接成分    |
|  g   |  语素字  |  x   | 非语素字 |  w   | 标点符号 |  ws  | 非汉字字符串 |  wu  | 其他未知的符号 |

### Process

The original corpus has been annotated and is of high quality, but for the program needs, some simple cleaning tasks were still performed, mainly to delete spaces.

```bash
sed -i 's/[][]//g' origin.txt             # Delete the ']' and '['.
sed -i -E 's/ +/ /g' origin.txt           # Delete repeated spaces.
sed -i 's/\s*\//\//g' origin.txt          # Delete spaces before the '/'.
sed -i -E 's/([^a-z]) +/\1/g' origin.txt  # Delete spaces after the non-alpha char.
```

## Datasets

### Spliting

Considering the GitHub's limitation on the single file's size and memory reading speed, it was decided to divide the original corpus file into multiple small files. And therefore...

```bash
split -l 200000 all.txt                   # Split large file to small files.
```

All splited files are stored under the `./corpus/train/` and named from 'xaa' to 'xal'.

Except for 'xal' which has 129263 lines, the other files are all 200000 lines and the average file size is about 45 MB.

Contrary, the test set is stored in the `./corpus/test` and named from 'yaa' to 'yak'.

All the test set are randomly generated from the set `train/xal`, which isn't used when training, with the shell command `shuf`.

```bash
shuf -n200 train/xal > test/yaa           # Generate the test data.
```

### Division

Regarding the division of the dataset, the time complexity of the training process and the testing process is mainly considered.

For every text line with the length $T$ and suppose that there are $N$ hidden states.

- Training traverses the sentence from beginning to end almost only once, so the time complexity is $O(T)$.
- While test uses Viterbi, which needs calculating the product between hidden states at each time, therefore costs $O(T \times N \times N)$.

The testing time is $N^2$ times the training time, so the data set should be divided according to this ratio.

In our case, $N$ is equal to 35, hence the ratio of test set to training set should be close to $1:1225$.

### Metadata

| train set |   file size    | test set |  file size  |
| :-------: | :------------: | :------: | :---------: |
| train/xaa | 200000L, 45.2M | test/yaa | 200L, 56.0K |
| train/xab | 200000L, 47.0M | test/yab | 200L, 57.8K |
| train/xac | 200000L, 47.5M | test/yac | 200L, 34.6K |
| train/xad | 200000L, 47.1M | test/yad | 200L, 36.9K |
| train/xae | 200000L, 47.3M | test/yae | 200L, 37.6K |
| train/xaf | 200000L, 44.9M | test/yaf | 200L, 43.3K |
| train/xag | 200000L, 43.6M | test/yag | 200L, 52.9K |
| train/xah | 200000L, 45.2M | test/yah | 200L, 60.0K |
| train/xai | 200000L, 45.1M | test/yai | 200L, 45.7K |
| train/xaj | 200000L, 43.9M | test/yaj | 200L, 53.2K |
| train/xak | 200000L, 43.7M | test/yak | 200L, 52.8K |

The package `common.dataset` provides a simple way to reference the datasets.

## Model

### Matrix

All the matrix are read from the origin corpus and saved as a json file.

And in the memory, they are stored in python dict, which can be read with $O(1)$.

The methods to read and save the 3 matrix are all implemented in the module `model.matrix`.

The metadata of the 3 matrix is described here.

```python
init_mt = {state:prob * 35}
tran_mt = {prev:{curr:prob * 35} * 35}
emis_mt = {state:{obs:prob * N} * 35}
```

The sizes of the three matrices extracted from the original corpus are: 

initial\_matrix 912 B;  transition\_matrix 28.9 K; emission\_matrix 7.64M;

### Algorithm

The model is based on the `Hidden Markov Model`, and caculated with the `Viterbi` algorithm.

Detailed information of this algorithm can be found at [this](https://en.wikipedia.org/wiki/Viterbi_algorithm).

This program's model is implemented with the module `model.markov` and `model.viterbi`.

The former realizes the generation and backtracking of the Markov chain, and the latter realizes the update of the Markov chain.

## Debugging

### Problem 1: 0-value

##### Analysis

During the debugging process, the `0-value problem` was mainly encountered, which is caused by the following aspects:

- Due to the sparsity of the corpus, there are extremely small values for transition probability and emission probability.
- Markov requires the cumulative multiplication of probabilities before and after, thus the probabilities continuously shrink during the period.
- When the sequence is too long, the probability may be lower than the smallest positive number that the computer can represent.

In addition to limitations in **computer representation capabilities**, 0-value problem also manifests itself in the **waste of available information**.

- Transition matrix and sparse matrix are actually sparse matrix containing a large number of 0 values.
- So the model encounters multiplication by 0 when calculating probabilities.
- This will cause that any subsequent operation will result in 0 and thus a waste of available information.

##### Solution

To solve the 0 value problem, the solution adopted is to replace the possible 0 value with a very small number.

In actual projects, `1e-99` was used.

Specifically, when encountering a word that is not in the corpus or the previous node's max probability of all states is a zero, set the probability to 1e-99 instead of 0.

### Problem 2: mq-tag

##### Analysis

In actual testing, problems with the original data set were also discovered. That is the tag 'mq', which does not appear in the metadata description.

- In the original data set, 'm' represents a numeral, 'q' represents a quantifier, and 'mq' does not exist.
- After investigation, 'mq' mainly marked words like **"一个"** and **"几个"**.
- The existence of 'mq' seriously reduces the accuracy of model prediction.

##### Solution

Replace the 'mq' to the 'm' when training and testing.

```python
if state == "mq":
    state = "m"
```

The impact of the change on the original model architecture is minimal, but the improvement in model accuracy is huge.

After adding 'mq', the accuracy of the model increased by about 0.44%.

## Experiments

### Parameters

The performance of the model is closely related to the selection of hyperparameters.

All hyperparameters can be found in the module `common.params`.

A total of three hyperparameters are defined in this project:

```python
MINIMAL_PROB = 1e-99
THRESHOULD = 1e-3
ENDSTATE_GUESS = "w"
MIDSTATE_GUESS = "n"
```

- **MINIMAL_PROB**: 
the minimum probability used to replace probability 0, used to solve the 0-value problem.

- **THRESHOULD**: 
if the probability of one of the end states is bigger than the sum of the probability of the state 'w' and the threshould, then use the state as the final state.

- **ENDSTATE_GUESS**: 
by default, guess the state of a final node is 'w' (punctuation mark).

- **MIDSTATE_GUESS**:
if the previous node's max probability of all the states is 0 then guess the previous node's hidden state is 'n' (noun).

```python
# find the end state with the highest probability

# by default guess the 'w'
end_state = ENDSTATE_GUESS
end_prob = markov_chain[-1][ENDSTATE_GUESS]["prob"]

for state in markov_chain[-1]:
    prob = markov_chain[-1][state]["prob"]
    # update if beyond the threshould
    if prob > end_prob + THRESHOULD:
        end_prob = prob
        end_state = state
```

### Results

##### Final Results

```
Last tested on 2023-10-18 00:40:40.062570.
Total number of test samples is 69290
There are 64225 correctly tagged and 5065 incorrectly.
Accuracy is 92.69%
Error prediction distribution:
n:1040	nt:69	nd:81	nl:33	nh:14	nhf:12	nhs:7	ns:74
nn:0	ni:14	nz:3	v:1122	vd:89	vl:115	vu:31	a:354
f:15	m:84	q:62	d:410	r:103	p:183	c:121	u:499
e:0	o:1	i:95	j:48	h:11	k:30	g:0	x:1
w:302	ws:42	wu:0	
```

The optimization of the model in this project is mainly based on the analysis of erroneous prediction samples.

`Optimization 1`

In the first round of experiments, the accuracy of the model reached 83.77%, which is a relatively high accuracy. When I delved into the error samples, I found that the model was most prone to misclassification for the marker 'w' (punctuation), with over 2000 errors.

After testing, it was found that this is due to the long test sentence and the small difference in the probabilities of each state of the final decision node. The model has a lower preference for the punctuation mark 'w' that should be located at the end of the sentence.

To address this issue, the hyperparameter **ENDSTATE_GUESS** and **THRESHOULD** has been introduced to add preferences to the model, making it guess 'w' by default, thereby improving accuracy.

`Optimization 2`

In the second round of experiments, it was noted that the model was prone to misclassification of 'n' (noun) and 'v' (verb), with more than 1000 misclassification times.

This is due to the incompleteness of the corpus itself. Even a medium-sized corpus cannot cover all language vocabulary, so the model feels 'unfamiliar' with these vocabulary.

To address this issue, the hyperparameter **MIDSTATE_GUESS** was introduced to guess unfamiliar words encountered during the execution of the Viterbi algorithm, making it default to guess 'n' and improving accuracy.

After testing, it has been found that the default guess 'v' can achieve a similar effect improvement. For general considerations, it is set to 'n'.

`Other`

After testing, adjusting **MINIMAL_PROBABILITY** to a smaller size will improve the accuracy of training, but it will bring about overfitting issues.

Especially when the sentence length is particularly long, the model will ultimately predict all words as 'n', which is the preset **MIDSTATE_GUESS**.

After comprehensive consideration, we still choose the larger 1e-99 as the **MINIMAL_PROBABILITY**.
