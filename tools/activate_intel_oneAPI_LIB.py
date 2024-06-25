import os

msvc_path = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\"
version_dir = os.listdir(msvc_path)

vc_version = version_dir[0]

LIB = (r"C:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\lib;"
       fr"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\{vc_version}\ATLMFC\lib\x64;"
       fr"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\{vc_version}\lib\x64;$env:LIB")

env_file = os.getenv('GITHUB_ENV')

with open(env_file, "a") as myfile2:
    myfile2.write(fr"LIB={LIB}")