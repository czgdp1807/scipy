import os

msvc_path = ("C:\\Program Files\\Microsoft Visual Studio\\2022\\"
             "Enterprise\\VC\\Tools\\MSVC\\")
version_dir = os.listdir(msvc_path)
vc_version = version_dir[-1]

PATH = ("C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        fr"Enterprise\VC\Tools\MSVC\{vc_version}\bin\HostX64\x64;"
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        r"Enterprise\Common7\IDE\VC\VCPackages;"
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        r"Enterprise\Common7\IDE\CommonExtensions\Microsoft\TestWindow;"
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        "Enterprise\\Common7\\IDE\\CommonExtensions\\Microsoft\\"
        r"TeamFoundation\Team Explorer;"
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        r"Enterprise\MSBuild\Current\bin\Roslyn;"
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        r"Enterprise\Team Tools\DiagnosticsHub\Collector;"
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\\x64;"
        r"C:\Program Files (x86)\Windows Kits\10\bin\\x64;"
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        r"Enterprise\\MSBuild\Current\Bin\amd64;"
        r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\;"
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\;"
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin;C:\Windows\system32;"
        r"C:\Windows;C:\Windows\System32\Wbem;"
        r"C:\Windows\System32\WindowsPowerShell\v1.0;"
        r"C:\Windows\System32\OpenSSH;"
        r"C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit;"
        r"C:\ProgramData\chocolatey\bin;"
        r"C:\Program Files\Git\cmd;"
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        r"Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;"
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        r"Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;"
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        r"Enterprise\Common7\IDE\VC\Linux\bin\ConnectionManagerExe;"
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
        r"Enterprise\VC\vcpkg;$env:PATH")

env_file = os.getenv('GITHUB_ENV')

with open(env_file, "a") as myfile1:
    myfile1.write(fr"PATH={PATH}")
