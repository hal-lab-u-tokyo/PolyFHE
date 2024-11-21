# Evaluation

## Memory Access 
```
make run
make run-noopt
./profile/profile-memaccess.sh
```
- This script will generate `data/memaccess-{opt/noopt}.csv` and `figure/memaccess.png` files.

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