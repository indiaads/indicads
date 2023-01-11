## Indian ads

To classify the images in a directory, set the ```feat_extractor :``` inside of model, in ```config.yaml``` to ```vit``` or ```convnext``` to use one of them as feature extractor, and then run ```classify.py``` by giving the arguments as following.

```
classify.py --dataset <path of dir with images> --ckpt_path <path of the ckpt file> --pred_out <predictions_file.pkl>
```
> NOTE: give a file name along with the .pkl or .pickle extension for --pred_out

Running the above script generate a .pkl file of a list of tuples, (image_filename, class_predicted).
