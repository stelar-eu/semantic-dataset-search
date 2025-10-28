## Installation 

The `front-end` showcase oriented package can be deployed using `uv`. Make sure you have installed it before proceeding. 

To run the code first run: 

```
uv venv
```

this will create a virtual environment for the project. 

To install the dependencies needed for the front-end only, run: 

```
uv sync --group front-end
```

Then you can run the streamlit front-end with: 

```
uv run --group front-end streamlit run front-end/Dashboard.py
```

The app takes as a given that there is an instance of the server running on `localhost:8000`