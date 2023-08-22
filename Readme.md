## Introduction

This is a header-only library. It does not need to be installed. Just clone the repo with

```bash
git clone https://github.com/SparksFlying/SDI-SOCKET.git
```

and copy the `include/pgm` directory to your system's or project's include path.
                                                                          
The `test/experiment.cpp` file shows how to run the environment what SDI needed, it contains four entities: DSP (i.e., $C_{1}$), DAP (i.e., $C_{2}$), CA, DO and AU.

1. Run CA via:
```
./experiment ca 1024
```

2. Run DAP via:
```
./experiment dap
```

3. Run DSP via:
```
./experiment dsp
```

4. Run DAP via:
```
./experiment dap
```

5. Run DO and choose the dataset (e.g., CAR, SYN or GOWALLA) with its size and dimension, then send the encrypted data and indexd file to DSP. For example, DO chooses 20000 two-dimensional points of CAR:
```
./experiment do car 2 20000
```

6. Run AU to send query requests to the server DSP via:
```
./experiment au car 2 20000 0.005
```
