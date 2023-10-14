# Time elastic band simple test with g2o

## Memo

* Time elastic band
* Using g2o and automatic diff
* Most code from `teb_local_planner`

## Build and run

```bash
git submodule update
cd third_party/g2o
git apply ../../g2o.patch
cd ../..
mkdir build
cd build
cmake ..
make
./teb
```