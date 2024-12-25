import onnxruntime as ort

session = ort.InferenceSession("cp/face_parser.onnx")

inputs = session.get_inputs()
for i, inp in enumerate(inputs):
    print(f"Input {i}:")
    print(f"  Name: {inp.name}")
    print(f"  Shape: {inp.shape}")
    print(f"  Type: {inp.type}")

outputs = session.get_outputs()
for i, out in enumerate(outputs):
    print(f"Output {i}:")
    print(f"  Name: {out.name}")
    print(f"  Shape: {out.shape}")
    print(f"  Type: {out.type}")

model_meta = session.get_modelmeta()
print("Model Metadata:")
print(f"  Name: {model_meta.graph_name}")
print(f"  Description: {model_meta.description}")
print(f"  Domain: {model_meta.domain}")
print(f"  Producer: {model_meta.producer_name}")