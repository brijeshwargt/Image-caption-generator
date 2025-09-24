# 📸 AI Image Caption Generator

A Streamlit web application that generates descriptive captions for uploaded images using deep learning.

## 🚀 Features

- **Image Upload**: Support for PNG, JPG, JPEG, GIF, BMP formats
- **AI Caption Generation**: Transformer-based architecture for natural language generation
- **Real-time Processing**: Fast image analysis (~2 seconds)
- **Professional UI**: Clean, responsive interface with custom styling
- **Technical Details**: Display image metrics and processing information

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow/Keras
- **Image Processing**: PIL, OpenCV
- **Architecture**: Transformer + CNN (InceptionV3)

## 📦 Installation

1. **Clone or download** this project
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 🌐 Deploy on Streamlit Cloud

1. **Push to GitHub** (include app.py and requirements.txt)
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Connect GitHub** and select your repository
4. **Set main file**: app.py
5. **Deploy** and get your public URL

## 📁 Project Structure

```
image_caption_generator/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎯 How It Works

1. **Image Upload**: User uploads an image through the web interface
2. **Preprocessing**: Image is resized to 299x299 and normalized
3. **Feature Extraction**: CNN extracts visual features from the image
4. **Caption Generation**: Transformer generates natural language description
5. **Display**: Results are shown with the original image

## 💡 Usage Tips

- **Best results**: Clear, well-lit images work best
- **Supported subjects**: People, animals, objects, scenes
- **Image quality**: Higher resolution generally produces better captions
- **File size**: Keep images under 10MB for best performance

## 🔧 Customization

The app can be extended with:
- Pre-trained model loading
- Multiple language support
- Batch processing capabilities
- Advanced model architectures
- Custom styling and themes

## 🚀 Deployment Notes

- **Python version**: Use 3.11 for best compatibility
- **Dependencies**: Minimal requirements for faster deployment
- **Memory**: App uses ~1GB RAM for TensorFlow
- **Performance**: Optimized for Streamlit Cloud limits

Built with ❤️ using Streamlit and TensorFlow
