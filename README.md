Dysgraphia Text Translator

A Streamlit web application that detects dysgraphic handwriting, extracts the handwritten text using OCR, and translates it into clear English using Google's Gemini AI.

## Features

- Detects handwriting regions using YOLO
- Extracts text with EasyOCR
- Identifies normal and dysgraphic handwriting
- Reconstructs dysgraphic text using Gemini AI
- User-friendly Streamlit interface

## Technologies Used

- Python
- Streamlit
- YOLO (Ultralytics)
- EasyOCR
- Google Gemini API
- OpenCV
- NumPy
- Pillow

## Project Workflow

1. Upload a handwriting image.
2. YOLO detects handwriting regions.
3. EasyOCR extracts the text.
4. The system determines whether the handwriting is normal or dysgraphic.
5. Gemini AI reconstructs dysgraphic text into readable English.
6. The translated result is displayed in the web interface.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Required Files

Place the trained YOLO model in the project directory:

```
YOLO.keras
```

Add your Gemini API key to Streamlit Secrets:

```
GEMINI_API_KEY = "YOUR_API_KEY"
```

## Run the Application

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py
├── YOLO.keras
├── requirements.txt
└── README.md
```

## Future Improvements

- Support multiple languages
- Improve OCR accuracy
- Add confidence scores
- Deploy as a public web application

## License

This project is intended for educational and research purposes.
