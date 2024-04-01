import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import spacy
from paddleocr import PaddleOCR
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import easyocr

# Définir les transformations pour l'image à prédire
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner l'image à la taille attendue de ResNet152
    transforms.Grayscale(num_output_channels=3),  # Convertir en image RVB si l'image est en niveaux de gris
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalisation des valeurs des pixels
])

# Charger le modèle
model = models.resnet152(pretrained=False)  # Charger le modèle ResNet152
num_classes = 32  # Remplacer ... par le nombre de classes de votre ensemble de données
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modifier la dernière couche

# Charger les poids du modèle pré-entraîné
model.load_state_dict(torch.load("/content/drive/MyDrive/resnet152_trained_model_vf.pth", map_location=torch.device('cpu')))
model.eval()  # Mettre le modèle en mode évaluation

# Fonction principale de l'application
def main():
    st.markdown("<h1 style='text-align: center; color: black;'>RIBs Parser</h1>", unsafe_allow_html=True)

    # Affichage de la zone de téléchargement de l'image
    uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Enregistrement de l'image téléchargée
        image_path = save_uploaded_file(uploaded_file)

        # Exécuter le traitement sur l'image
        process_image(image_path)

# Fonction pour enregistrer le fichier téléchargé
def save_uploaded_file(uploaded_file):
    # Créer un dossier de sortie s'il n'existe pas
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sauvegarder le fichier téléchargé
    image_path = os.path.join(output_dir, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return image_path

# Fonction pour le traitement de l'image
def process_image(image_path):
    # Charger l'image
    original_image = Image.open(image_path)

    # Charger le modèle YOLO avec les poids de votre modèle entraîné
    model_yolo = YOLO('/content/drive/MyDrive/best.pt')
    results = model_yolo([image_path])  # Exécuter l'inférence sur l'image

    # Process results list
    for result in results:
        # Get xyxy coordinates of bounding boxes for each detection
        boxes = result.boxes.xyxy

        # Iterate over each box
        for i in range(boxes.shape[0]):
            # Extract coordinates of the box
            x1, y1, x2, y2 = map(int, boxes[i, :4])  # Convert to integers

            # Crop the region of interest using PIL
            roi = original_image.crop((x1, y1, x2, y2))

            # Save the cropped region to the output directory
            output_path = os.path.join('output', f"cropped_{os.path.basename(image_path)}")
            roi.save(output_path)

    # Diviser l'espace en deux colonnes
    col1, col2 = st.columns(2)

    # Affichage de l'image originale dans la première colonne
    with col1:
        st.header('Image Originale')
        st.image(original_image, use_column_width=True)

    # Affichage de l'image résultante dans la deuxième colonne
    with col2:
        result_image_path = output_path  # Prendre le dernier chemin sauvegardé
        result_image = Image.open(result_image_path)
        st.header('Image Résultante')
        st.image(result_image, use_column_width=True)

    # Appliquer les transformations à l'image
    image_tensor = transform(result_image).unsqueeze(0)  # Ajouter une dimension de lot (batch) car le modèle s'attend à recevoir un lot d'images

    # Effectuer la prédiction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # Obtenir la classe prédite
    predicted_class = predicted.item()

    # Obtenir le nom de la classe prédite à partir de l'ensemble de données
    class_names = ["100°","10°","110°","120°","130°","140°","150°","160°","170°","190°","200°","20°","210°","220°","230°","240°","250°","260°","280°","290°","300°","30°","310°","320°","330°","340°","350°","40°","50°","60°","70°","80°"]  # Remplacer ... par les noms de classe de votre ensemble de données
    predicted_class_name = class_names[predicted_class]

    # Affichage de la classe prédite en gras
    st.markdown(f"Classe prédite pour {image_path} : <b>{predicted_class_name}</b>", unsafe_allow_html=True)

    angle_int = int(predicted_class_name[:-1])
    angle_int = -angle_int

    # Charger l'image à prédire
    image = Image.open(result_image_path)

    # Rotation de l'image
    rotated_img = image.rotate(angle_int, expand=True)

    # Chemin de sortie où sauvegarder l'image traitée
    output_image_path = "output/image_rotated.jpg"

    # Sauvegarder l'image
    rotated_img.save(output_image_path)
    print(f"Image rotated and saved to {output_image_path}")

    # Affichage de l'image traitée
    st.header('Image traitée')
    st.image(rotated_img, use_column_width=True)

    ocr = PaddleOCR(use_angle_cls=True,lang='fr')
    results = ocr.ocr(output_image_path)

    # Stocker le texte détecté dans une chaîne de caractères
    text = ''
    for result in results:
        for line in result:
            text += line[1][0] + ' ' # Le texte détecté
        text += '\n'


    # Affichage du texte détecté
    st.header("Texte détecté:")
    st.write(text)
    st.header("NER:")
    nlp = spacy.load("/content/drive/MyDrive/model-best")
    doc = nlp(text)
    for ent in doc.ents:
        st.write(ent.text, "->>>>>>>>", ent.label_)

if __name__ == "__main__":
    main()
