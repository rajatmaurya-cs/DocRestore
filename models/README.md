# Models Directory

This directory stores the trained Pix2Pix Generator model weights.

### Expected File
- **Path:** `models/generator.pth`
- **Format:** PyTorch state dictionary (either raw `state_dict` or a dictionary with key `generator_state_dict`).

### Integration Instructions
1. Train the model on Google Colab using the provided notebook `colab/DocRestore_Pix2Pix_Training.ipynb`.
2. When training completes, download `best_generator.pth` or `generator.pth`.
3. Rename or copy it to this folder as:
   ```bash
   models/generator.pth
   ```
4. Start or refresh the Streamlit app (`streamlit run app.py`). The app will automatically detect and load the trained model!
