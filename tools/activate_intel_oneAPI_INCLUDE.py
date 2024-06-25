import os

msvc_path = ("C:\\Program Files\\Microsoft Visual Studio\\2022\\"
             "Enterprise\\VC\\Tools\\MSVC\\")
version_dir = os.listdir(msvc_path)
vc_version = version_dir[-1]

INCLUDE = ("C:\\Program Files\\Microsoft Visual Studio\\2022\\"
           fr"Enterprise\VC\Tools\MSVC\{vc_version}\include;"
           "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
           fr"Enterprise\VC\Tools\MSVC\{vc_version}\ATLMFC\include;"
           "C:\\Program Files\\Microsoft Visual Studio\\2022\\"
           r"Enterprise\VC\Auxiliary\VS\include;"
           r"C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\ucrt;"
           r"C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\um;"
           r"C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\shared;"
           r"C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\winrt;"
           r"C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0"
           r"\\cppwinrt;$env:INCLUDE")

env_file = os.getenv('GITHUB_ENV')

with open(env_file, "a") as myfile3:
    myfile3.write(fr"INCLUDE={INCLUDE}")
