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
    UV_PYTHON = "${pkgs.python312}/bin/python3";
    UV_PYTHON_DOWNLOAD = "never";
  };
}
