# Handle WindowsPath on non-Windows systems
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# Model path
model_path = 'transport_model.pkl'

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file {model_path} does not exist.")
else:
    st.success(f"Model file {model_path} found.")

# Initialize the model variable
model = None

# Try loading the model with fastai
try:
    model = load_learner(model_path)
    st.success("Model loaded successfully with fastai.")
except Exception as e:
    st.error(f"Error loading model with fastai: {e}")

# Streamlit app title
st.title("Transportni klassifikatsiya qiluvchi model")

# File uploader for images
file = st.file_uploader("Rasm yuklash", type=['png', 'jpg', 'gif', 'svg'])
if file:
    st.image(file)
    
    # Convert uploaded file to PILImage
    img = PILImage.create(file)
    
    # Check if the model is loaded before making a prediction
    if model:
        # Predict using the model
        pred, pred_id, probs = model.predict(img)
        
        # Display the prediction and probability
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
        
        # Plotting the probabilities
        fig = px.bar(x=probs*100, y=model.dls.vocab, labels={'x': 'Ehtimollik', 'y': 'Sinf'})
        st.plotly_chart(fig)
    else:
        st.error("Model is not loaded. Cannot make predictions.")