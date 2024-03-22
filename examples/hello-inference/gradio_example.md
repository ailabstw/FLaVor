# Visualize Your Inference Output with Gradio Strategy

![Gradio example](./images/gradio_example.png)

To quickly evaluate how well the model performs, we provide a visualization tool using [Gradio](https://github.com/gradio-app/gradio), an open-source Python package that allows for the fast deployment of machine learning models with a GUI. Currently, we support the following tasks:

* Segmentation task with `GradioSegmentationStrategy`
* Detection task with `GradioDetectionStrategy`

## Prerequisite

Please follow the instructions in [segmentation example](./segmentation_example.md).

## Getting started

Let's demonstrate Gradio Strategy with [segmentation example](./segmentation_example.md). To adopt Gradio interface, we provide base class in `base_gradio_inference_model.py`. By reusing most of the code in `seg_example.py`, we can demonstrate the usage of Gradio service in `gradio_example.py`.

Here's the output format for gradio inference model in segmentation task. You can see that we only add a new key-value pair `data` in the `infer_output`.

```python
infer_output = {
    "sorted_images": [{"id": uid, "file_name": file_name, "index": index, ...}, ...],
    "categories": {class_id: {"name": category_name, "supercategory_name": supercategory_name, display: True, ...}, ...},
    "model_out": model_out, # 3d/4d NumPy array with segmentation predictions.
    "data": data # Original input data. Must be specified for Gradio
}
```

Then, use `GradioInferAPP` by specifying `GradioInputStrategy` for `input_strategy` and `GradioSegmentationStrategy` for `output_strategy` as follows:

```python
app = GradioInferAPP(
    infer_function=GradioSegmentationStrategy(),
    input_strategy=GradioInputStrategy,
    output_strategy=GradioSegmentationStrategy,
)
```

By using `GradioInputStrategy`, you won't need additional AiCOCO input anymore.

Initiate the service as given in the examples, and go to <http://localhost:9000/>. You are all set! Play around with the Gradio GUI using your inference model.
