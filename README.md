## TinyCV
This header-only C++14 library is intended to contain implementations of
advanced computer vision algorithms.

The next sections describe the currently implemented algorithms.

### Mutual Information Registration
Performs mutual information registration between two images, as described in
the work *Accurate Real-time Tracking Using Mutual Information* by Dame et al.

## <a name="requirements"></a> Requirements
* A C++14 compliant compiler (GCC 5 or newer is recommended)
* CMake 3.1 or newer
* git for cloning the repository

## Building
First, make sure your system satisfy the [minimum requirements](#requirements).

After cloning the repository, make sure to initialize its submodules :

    git submodule init
    git submodule update

Finally, create a *build* folder in any place you like and run the following
commands *inside* it:

```shell
cmake <path/to/cloned/repository>
make -j8
```

## Documentation

## References
1. [Accurate Real-time Tracking Using Mutual Information][1]

[1]: https://www.irisa.fr/lagadic/pdf/2010_ismar_dame.pdf
