# Test models

Each directory has ONNX test models and the Python script that generates them.

To generate the models of a script, run the script in its directory, either with
`uv run <script>.py`, or in a virtual environment in a temporary directory:

```sh
python3 -m venv /tmp/onnx-venv
/tmp/onnx-venv/bin/pip install onnx==1.23.2 numpy
/tmp/onnx-venv/bin/python <script>.py
```
