# Evaluation

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