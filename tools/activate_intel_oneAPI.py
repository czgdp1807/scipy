import os

msvc_path = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\"
version_dir = os.listdir(msvc_path)

vc_version = version_dir[0]

PATH = (r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin;"
        fr"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\{vc_version}\bin\HostX64\x64;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\VC\VCPackages;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\TestWindow;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\bin\Roslyn;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Team Tools\DiagnosticsHub\Collector;"
        r"C:\Program Files (x86)\Windows Kits\10\bin\\x64;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\\MSBuild\Current\Bin\amd64;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\;"
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin;C:\Windows\system32;"
        r"C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0;"
        r"C:\Windows\System32\OpenSSH;C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit;"
        r"C:\ProgramData\chocolatey\bin;C:\Program Files\Git\cmd;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\VC\Linux\bin\ConnectionManagerExe;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\vcpkg;$env:PATH")

LIB = (r"C:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\lib;"
       fr"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\{vc_version}\ATLMFC\lib\x64;"
       fr"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\{vc_version}\lib\x64;$env:LIB")

INCLUDE = (fr"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\{vc_version}\include;"
           fr"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\{vc_version}\ATLMFC\include;"
           r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\VS\include;$env:INCLUDE")

env_file = os.getenv('GITHUB_ENV')

with open(env_file, "a") as myfile:
    myfile.write(fr"PATH={PATH}")
    myfile.write(fr"LIB = {LIB}")
    myfile.write(fr"INCLUDE = {INCLUDE}")