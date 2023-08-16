## Introduction

This is a header-only library. It does not need to be installed. Just clone the repo with

```bash
git clone https://github.com/SparksFlying/SDI-SOCKET.git
```

and copy the `include/pgm` directory to your system's or project's include path.
                                                                          
The `test/experiment.cpp` file shows how to run the environment what SDI needed, it contains four entities: DSP,DAP,CA,DO and AU.

<img src="" alt="system model" style="width: 300px">

1. Run ca via:
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

5. Run DO and choose dataset(car,syn or gowalla) with its size and dimension, then send the encrypted data and indexd file to DSP, for example DO chooses 20000 two-dimensional points of car:
```
./experiment do car 2 20000
```

6. Run AU to make query requests to the server DSP via:
```
./experiment au car 2 20000 0.005
```
. The '0.005' denotes the selectivity of queries.