{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    ruff
    uv
  ];

  env = {
    # uv wheels need shared libraries that NixOS does not expose in /usr/lib.
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
      with pkgs;
      [
        stdenv.cc.cc.lib
        zlib
        libxcb
        libGL
        glib
      ]
    );
    UV_PYTHON = "${pkgs.python312}/bin/python3";
    UV_PYTHON_DOWNLOADS = "never";
  };
}
