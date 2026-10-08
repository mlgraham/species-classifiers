"""Run a .tflite file with the same calling convention as a Keras model, so classify.py and webcam.py
accept either. Input is uint8 or float32 RGB in the 0..255 range, shape [N, H, W, 3]; output is a float
probability array [N, classes] (uint8 outputs are dequantized with the tensor's scale and zero point)."""

import numpy as np


class TFLiteModel:
    def __init__(self, path, threads=4):
        import tensorflow as tf

        self.interp = tf.lite.Interpreter(model_path=path, num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.input_shape = tuple(int(d) for d in self.inp["shape"])

    def predict(self, batch, verbose=0):
        batch = np.asarray(batch)
        scale, zero = self.out["quantization"]
        results = []
        for x in batch:
            self.interp.set_tensor(self.inp["index"], x.astype(self.inp["dtype"])[None])
            self.interp.invoke()
            y = self.interp.get_tensor(self.out["index"])[0].astype(np.float32)
            if scale:
                y = (y - zero) * scale
            results.append(y)
        return np.stack(results)


def load_any(path, threads=4):
    """A Keras .keras model or a .tflite file, with .predict() and .input_shape."""
    if path.endswith(".tflite"):
        return TFLiteModel(path, threads)
    import tensorflow as tf

    return tf.keras.models.load_model(path)
