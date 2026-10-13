{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    ruff
    uv

    # C++ 数据准备工具的构建驱动；编译器与 libc++ 来自 MaaDeps 自带 toolchain。
    ccache
    cmake
    ninja

    openssl
  ];

  shellHook = ''
    # MaaDeps 的 vcpkg 前缀不在 Nix 的标准查找路径里；显式导出以绕过 MaaDeps toolchain
    # 被判为交叉编译时 CMAKE_FIND_ROOT_PATH 与 CMAKE_PREFIX_PATH 的分叉（见票 01）。
    export CMAKE_PREFIX_PATH="${toString ./.}/cpp/MaaEnd/agent/cpp-algo/MaaUtils/MaaDeps/vcpkg/installed/maa-x64-linux''${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  '';

  env = {
    # uv wheels need shared libraries that NixOS does not expose in /usr/lib.
    LD_LIBRARY_PATH = "${
      pkgs.lib.makeLibraryPath (
        with pkgs;
        [
          stdenv.cc.cc.lib
          zlib
          libxcb
          libGL
          glib
        ]
      )
    }:${toString ./.}/cpp/MaaEnd/agent/cpp-algo/MaaUtils/MaaDeps/runtime/maa-x64-linux:${toString ./.}/cpp/build/bin";
    UV_PYTHON = "${pkgs.python312}/bin/python3";
    UV_PYTHON_DOWNLOADS = "never";
    C_INCLUDE_PATH = "${pkgs.linuxHeaders}/include";
  };
}
