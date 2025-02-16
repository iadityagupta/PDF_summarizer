# PDF Summarizer

A PDF summarizer using the [LaMini-Flan-T5-248M model](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M) and the Streamlit framework for the frontend.

## How to Set Up

1. **Create a Virtual Environment**
   ```python
   python -m venv venv
   ```
   
   
2. **Install the Required Libraries**
   ```python
   pip install -r requirements.txt
   ```
   
   
3. **Download the Model Files**

    Download all files for the [LaMini-Flan-T5-248M model](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M) model and save them in your project folder.


   
5. **Update the Checkpoint in app.py**

    Modify the checkpoint path in app.py to match the location of the model files in your project folder.


   
7. **Create an offload_folder**
 
   Create a directory named offload_folder in your project folder.
   


8. **Run the Application**
 
   ```python
   streamlit run app.py
   ```


You may also change the length of summarizing paragraph by changing the code in the app.py
