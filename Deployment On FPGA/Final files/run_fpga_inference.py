import vart, xir, numpy as np, time, os

def run_dpu_inference(xmodel_path, input_file):
    graph = xir.Graph.deserialize(xmodel_path)
    subgraph = [sg for sg in graph.get_root_subgraph().children if sg.has_attr("device")][0]
    runner = vart.Runner.create_runner(subgraph, "run")

    input_data = np.load(input_file).astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    output_tensors = runner.get_output_tensors()
    output_shape = tuple(output_tensors[0].dims)
    output_data = np.zeros(output_shape, dtype=np.float32)

    start = time.time()
    job_id = runner.execute_async([input_data], [output_data])
    runner.wait(job_id)
    elapsed = (time.time() - start) * 1000

    pred = np.argmax(output_data)
    label = "🔫 Gunshot" if pred == 1 else "🌳 Non-Gunshot"
    print(f"Prediction: {label}, Inference time: {elapsed:.2f} ms")

if __name__ == "__main__":
    run_dpu_inference("./compiled_model/gunshot_embedded.xmodel", "test_sample.npy")
