# Running a custom model with Slimp

Using the files in this directory, and assuming that Slimp is installed in `$HOME/local`, a custom model can be run as such:

```bash
mkdir build
cd build
CXXFLAGS="-I $HOME/local/include" LDFLAGS="-L $HOME/local/lib" cmake ../
make VERBOSE=1
python3 ../run_model.py
```
