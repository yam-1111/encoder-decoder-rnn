# Laboratory Activity

Encoder–Decoder RNN Laboratory

**Mission:** Hanapin Mo Ang Bit

## Objectives

At the end of the activity, students should be able to:

1. explain how an Encoder–Decoder RNN processes an input sequence and generates an output sequence;

2. implement a simple sequence-to-sequence model using Python without libraries except only NumPy;

3. compare model behavior across four different sequence tasks: reversal, addition, Caesar cipher decoding,
and echo-with-parity.


## Materials

Each group will need:

- 1 laptop or desktop computer
- Python environment with NumPy library
- code editor or notebook environment

## Uniform Lab Constraints

To make the difficulty level consistent across all groups, all teams must follow these rules:

- use an Encoder–Decoder RNN;
- use SGD as the optimizer;
- use a fixed learning rate of 0.01;
- use one-hot encoding for all inputs.


### Part A: Scenario

A dance trend is currently popular that keeps appearing repeatedly on TikTok and IG Reels. But because of the
many reposts, edits, and remixes, some sequences have been cut, rearranged, or have missing pieces of the signal.
In the mission “Hanapin Mo Ang Bit,” you are the coding team assigned to restore the correct pattern behind the
viral beat. You will use an Encoder–Decoder RNN to read the broken input sequence and reconstruct the correct
output, like trying to find the right moves behind the repeating music.


### Part B: Group Assignments

Each group will solve one protocol.

#### Group 1: "'Mirror Mirror' Protocol"

**Task:** The model reads a sequence of digits and outputs the exact reverse. Example: input [4, 1, 9] → output [9, 1,
4]. This task emphasizes strong dependence on the context vector, since the decoder must reconstruct the sequence
in reverse order.

#### Group 2: "Secret Accountant"

**Task:** The encoder reads a string such as “15+08” and the decoder outputs the sum “23”. This task adds difficulty
because the sequence length may vary, and the decoder must learn when to stop.

#### Group 3: “Broken Typewriter”

**Task:** The encoder reads encrypted text produced by a Caesar shift, and the decoder reconstructs the original word.
Example: encrypted G-R-J corresponds to original D-O-G. This task has higher dimensionality because the vocabulary uses the alphabet A–Z.

#### Group 4: The Liar Detector”

**Task:** The encoder reads a binary string, and the decoder repeats the sequence and appends a parity bit: output 1 if the number of 1s is odd, and 0 if even. Example: input 1-0-1 → output 1-0-1-0 if the count of 1s is even, or 1-0-1-1 if odd, depending on the sequence. This task checks whether the model can remember the full sequence and compute parity

### Part C: Data Preperation

Each group prepares a small synthetic dataset based on its assigned task.

- Group 1: digit sequences using vocabulary size 10 (0–9), fixed length
- Group 2: characters 0–9, plus +, plus pad, variable length up to 5
- Group 3: uppercase alphabet A–Z, fixed length
- Group 4: binary digits 0 and 1, fixed length

All sequences must be converted to ***one-hot vectors** before being fed into the encoder.

### Part D: Model Construction

Each group builds the same general seq2seq pipeline:
1. Encoder
–
 reads the input sequence one symbol at a time;
–
 updates its hidden state at every time step;
–
 passes the final hidden state as the context vector.
2. Decoder
–
 starts from the context vector;
–
 predicts one output token at a time;
–
 continues until the full target sequence is produced.
3. Training
–
 compute loss for predicted vs. target output;
–
 update parameters using SGD with learning rate 0.01.

### Part E: Experiment Paper

Each group conducts the experiment by training and testing its model on the assigned protocol.


#### Group 1 Procedure
- generate several digit sequences of fixed length;
- train the model to reverse them;
- test on unseen digit sequences;
- check whether the decoder reproduces the exact reverse order.

#### Group 2 Procedure
- generate addition strings such as 03+05, 11+07, 15+08;
- train the model to output the correct sum;
- test on unseen examples;
- observe if the model learns both arithmetic structure and stopping condition.

#### Group 3 Procedure
- generate encrypted words using a fixed Caesar shift;
- train the model to reconstruct the original words;
- test on new encrypted words;
- observe whether the model learns the shift relationship.

#### Group 4 Procedure
- generate binary strings;
- train the model to echo the input and append the parity bit;
- test on unseen strings;
- observe whether the model preserves the original sequence and computes parity correctly.


### Part F: Monitoring and Visualization

Each group must prepare:
- a Loss vs. Iteration plot;
- a heatmap of the hidden state.
These visualizations help show how the hidden state changes while the encoder reads the sequence.

### Part G. Analysis Questions
Each group should answer:
1. Did the model produce the correct output on test examples?
2. How fast did the loss decrease?
3. What patterns appeared in the hidden-state heatmap?
4. What does the result suggest about the strengths and limits of Encoder–Decoder RNNs?

## Deliverables

Each group should submit:

- source code;
- sample training and test outputs;
- Loss vs. Iteration graph;
- hidden-state heatmap;
- short discussion of results.