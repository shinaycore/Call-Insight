{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python311Full
    pkgs.python311Packages.numpy
    pkgs.gcc        # <-- this provides libstdc++ needed by NumPy
    pkgs.unzip
  ];

  shellHook = ''
  # auto-activate venv if present
  if [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
  fi

  # add libstdc++ to LD_LIBRARY_PATH so Python C extensions work
  export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH
'';

}
