# extract-method-generation

## CC Dependency

- First force purge all existing modules. Use this command `module --force purge`
- Load the `StdEnv/2023` since it comes bundled with latest `gcc (gcc/12.3)` and `python 3.11`. Load it using `module load StdEnv/2023`
- Load `arrow` using `module load arrow/14.0.1`
- To check for available versions for any module and how to load them use this template `module spider <module_name>`
- Create `venv` in python. If you are facing difficulties in installing libraries with in-built dependency of let's say arrow, use `pip install --no-index <pkg_name>`