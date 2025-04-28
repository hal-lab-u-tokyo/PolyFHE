# Evaluation

## Memory Access 
```
make run
make run-noopt
./profile/profile-memtransfer.sh
```
- This script will generate `data/memtransfer-{opt/noopt}.csv` and `figure/memtransfer.png` files.

## Stall Reason
- For optimized build
```
make run
./profile/profile-stallreason.sh opt
```

- For non-optimized build
```
make run-noopt
./profile/profile-stallreason.sh noopt
```
- This script will generate `data/{opt/noopt}.csv` and `figure/stallreason-{opt/noopt}.png` files.