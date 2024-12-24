# About
This section talks about
- Data Augmentation and Cleaning
- Deploying to PROD

# Deployment Steps
1. Go to the Hugging Face website
2. Create a space in Gradio
3. `clone` that project then follow the instructions on the site
4. `export` the model that you've trained
5. Setup your `app.py` using `load_model.ipynb` as a reference
    - Make sure to add your requirements.txt, example images, and model
    - Make sure to use git lfs for the model since it is too big to be tracked by git
    - Reference: https://www.tanishq.ai/blog/posts/2021-11-16-gradio-huggingface.html

### Usability
- aside from having a UI to sanity check your model, Hugging Face also has a built-in API for your model
- this means you can create any frontend and just make API calls to the Hugging Face model API
