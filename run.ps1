# PowerShell launcher for EduPath AI
$env:PYTHONPATH = ".\.venv\Lib\site-packages"
& "C:\Users\SHAREEF\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" -m streamlit run app.py --server.port 8501
